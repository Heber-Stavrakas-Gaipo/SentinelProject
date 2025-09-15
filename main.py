import cv2
import numpy as np
import face_recognition
import os
import re
from ultralytics import YOLO
import pytesseract
from collections import deque, Counter

# --- CLASSE AUXILIAR PARA ESTABILIZAÇÃO DE TEXTO ---
# Esta classe irá armazenar as últimas leituras de OCR e retornar a mais estável.
class TextStabilizer:
    def __init__(self, buffer_size=15, min_len=7):
        self.buffer = deque(maxlen=buffer_size)
        self.stable_text = ""
        self.min_len = min_len

    def update(self, new_text):
        """Adds a new read text to the buffer if it has the minimum length."""
        if new_text and len(new_text) >= self.min_len:
            self.buffer.append(new_text)

    def get_stable_text(self):
        """Returns the most common text in the buffer if it is predominant."""
        if not self.buffer:
            return self.stable_text # Return last stable value if buffer is empty
        counts = Counter(self.buffer)
        most_common_text, count = counts.most_common(1)[0]
        # Only update if the most common reading appears in at least 30% of the buffer
        if count / len(self.buffer) > 0.3:
            self.stable_text = most_common_text
        return self.stable_text

# --- CONFIGURAÇÃO INICIAL ---
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

try:
    yolo_plate_model = YOLO('license_plate_detector.pt')
    print("Modelo YOLO de detecção de placas carregado com sucesso.")
except Exception as e:
    print(f"Erro ao carregar o modelo YOLO 'license_plate_detector.pt': {e}")
    exit()

# --- FUNÇÕES DE BANCO DE DADOS E ALERTA (Sem alterações) ---

def load_known_faces(path='./database/faces'):
    known_encodings, known_names = [], []
    print("Loading face database...")
    for filename in os.listdir(path):
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            try:
                image_path = os.path.join(path, filename)
                image = face_recognition.load_image_file(image_path)
                encoding = face_recognition.face_encodings(image)[0]
                known_encodings.append(encoding)
                known_names.append(os.path.splitext(filename)[0])
            except (IndexError, Exception) as e:
                print(f"WARNING: Error processing {filename}: {e}")
    print(f"{len(known_names)} faces loaded.")
    return known_encodings, known_names


def load_suspicious_plates(path='./database/suspicious_plates.txt'):
    print("Loading plate database...")
    try:
        with open(path, 'r') as f:
            plates = [clean_plate_text(line) for line in f]
        print(f"{len(plates)} plates loaded.")
        return plates
    except FileNotFoundError:
        print(f"WARNING: Plate file '{path}' not found.")
        return []


def trigger_alert(message, frame, box_coords=None):
    print(f"ALERT: {message}")
    if box_coords:
        (startX, startY, endX, endY) = box_coords
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 3)
        cv2.putText(frame, "ALERT:", (box_coords[0], startY - 10),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 2)
    # Show main alert message at the top of the screen
    cv2.putText(frame, message, (10, 30),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 255), 2)



def clean_plate_text(text):
    return re.sub(r'[^A-Z0-9]', '', text).upper()

# --- FUNÇÃO DE PROCESSAMENTO PRINCIPAL (ATUALIZADA) ---

def process_frame(frame, known_encodings, known_names, suspicious_plates, plate_stabilizer):
    # 1. FACE RECOGNITION
    rgb_small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)[:, :, ::-1]
    rgb_small_frame = np.ascontiguousarray(rgb_small_frame)
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
        name = "Unknown"
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_names[best_match_index]
                top, right, bottom, left = [c * 2 for c in face_location]
                trigger_alert(f"Suspicious face: {name}", frame, (left, top, right, bottom))

    # 2. LICENSE PLATE RECOGNITION PIPELINE
    plate_results = yolo_plate_model(frame, verbose=False)

    current_frame_plates = []
    for result in plate_results:
        for box in result.boxes:
            x1, y1, x2, y2 = [int(i) for i in box.xyxy[0]]

            # a. Perspective Correction
            src_pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype="float32")
            width, height = 400, 120
            dst_pts = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warped_plate = cv2.warpPerspective(frame, M, (width, height))
            gray = cv2.cvtColor(warped_plate, cv2.COLOR_BGR2GRAY)

            # b. Multi-Strategy Preprocessing
            # Strategy A: Contrast
            sharpened = cv2.addWeighted(gray, 2.5, cv2.GaussianBlur(gray, (0, 0), 3.0), -1.5, 0)
            _, ocr_image_A = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            if np.mean(ocr_image_A) < 128: ocr_image_A = cv2.bitwise_not(ocr_image_A)

            # Strategy B: Illumination
            denoised_bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
            ocr_image_B = cv2.adaptiveThreshold(denoised_bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            if np.mean(ocr_image_B) > 128: ocr_image_B = cv2.bitwise_not(ocr_image_B)

            # c. Reading and Selection
            try:
                custom_config = r'--oem 3 --psm 7 -l por -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                plate_A = clean_plate_text(pytesseract.image_to_string(ocr_image_A, config=custom_config))
                plate_B = clean_plate_text(pytesseract.image_to_string(ocr_image_B, config=custom_config))

                ideal_length = 7
                len_A, len_B = len(plate_A), len(plate_B)

                clean_plate = ""
                if len_A == ideal_length and len_B != ideal_length:
                    clean_plate = plate_A
                elif len_B == ideal_length and len_A != ideal_length:
                    clean_plate = plate_B
                else:
                    clean_plate = plate_A if len_A >= len_B else plate_B

                # d. Update the stabilizer with the best result from the frame
                plate_stabilizer.update(clean_plate)
                current_frame_plates.append({'box': (x1, y1, x2, y2)})

            except Exception:
                continue

    # e. Draw the stable result on the screen
    stable_plate_text = plate_stabilizer.get_stable_text()
    for plate_info in current_frame_plates:
        (x1, y1, x2, y2) = plate_info['box']
        box_color = (255, 0, 0) # Blue

        if stable_plate_text and stable_plate_text in suspicious_plates:
            box_color = (0, 0, 255) # Red
            trigger_alert(f"Plate: {stable_plate_text}", frame, (x1, y1, x2, y2))

        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        cv2.putText(frame, stable_plate_text, (x1, y1 - 15), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 0), 5)
        cv2.putText(frame, stable_plate_text, (x1, y1 - 15), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)

# --- FUNÇÃO MAIN (sem alterações) ---

def main():
    known_encodings, known_names = load_known_faces()
    suspicious_plates = load_suspicious_plates()
    video_capture = cv2.VideoCapture(0)

    plate_stabilizer = TextStabilizer()

    if not video_capture.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        process_frame(frame, known_encodings, known_names, suspicious_plates, plate_stabilizer)
        cv2.imshow('Surveillance System', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()