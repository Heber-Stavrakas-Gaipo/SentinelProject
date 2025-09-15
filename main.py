import cv2
import numpy as np
import face_recognition
import os
import re
from ultralytics import YOLO
import pytesseract
from collections import deque, Counter

# --- CLASSE AUXILIAR PARA ESTABILIZAÇÃO DE TEXTO ---
# (Mantida, pois é essencial para uma boa experiência de usuário)
class TextStabilizer:
    def __init__(self, buffer_size=15, min_len=7):
        self.buffer = deque(maxlen=buffer_size)
        self.stable_text = ""
        self.min_len = min_len

    def update(self, new_text):
        if new_text and len(new_text) >= self.min_len:
            self.buffer.append(new_text)

    def get_stable_text(self):
        if not self.buffer:
            return self.stable_text
        counts = Counter(self.buffer)
        most_common_text, count = counts.most_common(1)[0]
        if count / len(self.buffer) > 0.3:
            self.stable_text = most_common_text
        return self.stable_text

# --- CONFIGURAÇÃO INICIAL ---
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

try:
    # <<< MELHORIA: Carregar ambos os modelos necessários >>>
    # Modelo para detecção de veículos (carros, ônibus, motos)
    yolo_vehicle_model = YOLO('yolov8n.pt') 
    # Modelo especializado para detecção de placas
    yolo_plate_model = YOLO('license_plate_detector.pt')
    print("Modelos YOLO carregados com sucesso.")
except Exception as e:
    print(f"Erro ao carregar modelos YOLO: {e}")
    exit()
    
# Mapeamento de classes do COCO para veículos
VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck

# --- FUNÇÕES DE BANCO DE DADOS E ALERTA (sem alterações) ---
def load_known_faces(path='./database/faces'):
    known_encodings, known_names = [], []
    print("Carregando banco de dados de rostos...")
    for filename in os.listdir(path):
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            try:
                image_path = os.path.join(path, filename)
                image = face_recognition.load_image_file(image_path)
                encoding = face_recognition.face_encodings(image)[0]
                known_encodings.append(encoding)
                known_names.append(os.path.splitext(filename)[0])
            except (IndexError, Exception) as e:
                print(f"AVISO: Erro ao processar {filename}: {e}")
    print(f"{len(known_names)} rostos carregados.")
    return known_encodings, known_names

def load_suspicious_plates(path='./database/suspicious_plates.txt'):
    print("Carregando banco de dados de placas...")
    try:
        with open(path, 'r') as f:
            plates = [clean_plate_text(line) for line in f]
        print(f"{len(plates)} placas carregadas.")
        return plates
    except FileNotFoundError:
        print(f"AVISO: Arquivo de placas '{path}' não encontrado.")
        return []

def trigger_alert(message, frame, box_coords=None):
    print(f"ALERTA: {message}")
    if box_coords:
        (startX, startY, endX, endY) = box_coords
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 3)
        cv2.putText(frame, "ALERTA:", (box_coords[0], startY - 10),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(frame, message, (10, 30),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 255), 2)

def clean_plate_text(text):
    return re.sub(r'[^A-Z0-9]', '', text).upper()

# --- FUNÇÃO DE PROCESSAMENTO PRINCIPAL (REESTRUTURADA) ---
def process_frame(frame, known_encodings, known_names):
    """Processa um único frame para encontrar rostos e placas. Retorna os resultados."""
    face_results = []
    plate_results_data = []

    # --- 1. RECONHECIMENTO FACIAL ---
    # Redimensiona o frame para o reconhecimento facial (mais rápido)
    rgb_small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)[:, :, ::-1]
    rgb_small_frame = np.ascontiguousarray(rgb_small_frame)
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    
    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
        name = "Desconhecido"
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_names[best_match_index]
        
        top, right, bottom, left = [c * 2 for c in face_location] # Ajusta as coordenadas para o frame original
        face_results.append({'box': (left, top, right, bottom), 'name': name})

    # --- 2. PIPELINE DE DETECÇÃO EM CASCATA: VEÍCULO -> PLACA -> OCR ---
    # a. Detectar veículos no frame completo
    vehicle_detections = yolo_vehicle_model(frame, classes=VEHICLE_CLASSES, verbose=False)

    for vehicle in vehicle_detections[0].boxes:
        vx1, vy1, vx2, vy2 = [int(i) for i in vehicle.xyxy[0]]
        # Recortar a região de interesse (ROI) do veículo
        vehicle_roi = frame[vy1:vy2, vx1:vx2]

        # b. Detectar placas SOMENTE dentro da ROI do veículo
        plate_detections = yolo_plate_model(vehicle_roi, verbose=False)
        for plate in plate_detections[0].boxes:
            # Coordenadas da placa relativas à ROI do veículo
            px1_rel, py1_rel, px2_rel, py2_rel = [int(i) for i in plate.xyxy[0]]
            # Converter para coordenadas absolutas no frame original
            px1_abs, py1_abs = vx1 + px1_rel, vy1 + py1_rel
            px2_abs, py2_abs = vx1 + px2_rel, vy1 + py2_rel

            # c. Correção de Perspectiva e OCR (lógica validada)
            src_pts = np.array([[px1_abs, py1_abs], [px2_abs, py1_abs], [px2_abs, py2_abs], [px1_abs, py2_abs]], dtype="float32")
            width, height = 400, 120
            dst_pts = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warped_plate = cv2.warpPerspective(frame, M, (width, height))
            gray = cv2.cvtColor(warped_plate, cv2.COLOR_BGR2GRAY)

            # d. Multi-Estratégia de Pré-processamento e Seleção
            # (Código validado da análise estática)
            try:
                sharpened = cv2.addWeighted(gray, 2.5, cv2.GaussianBlur(gray, (0,0), 3.0), -1.5, 0)
                _, ocr_image_A = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                if np.mean(ocr_image_A) < 128: ocr_image_A = cv2.bitwise_not(ocr_image_A)

                denoised_bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
                ocr_image_B = cv2.adaptiveThreshold(denoised_bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
                if np.mean(ocr_image_B) > 128: ocr_image_B = cv2.bitwise_not(ocr_image_B)

                config = r'--oem 3 --psm 7 -l por -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                placa_A = clean_plate_text(pytesseract.image_to_string(ocr_image_A, config=config))
                placa_B = clean_plate_text(pytesseract.image_to_string(ocr_image_B, config=config))

                len_A, len_B = len(placa_A), len(placa_B)
                clean_plate = ""
                if len_A == 7 and len_B != 7: clean_plate = placa_A
                elif len_B == 7 and len_A != 7: clean_plate = placa_B
                else: clean_plate = placa_A if len_A >= len_B else placa_B
                
                plate_results_data.append({'box': (px1_abs, py1_abs, px2_abs, py2_abs), 'text': clean_plate})
            except Exception:
                continue
    
    return face_results, plate_results_data

# --- FUNÇÃO MAIN (ATUALIZADA PARA OTIMIZAÇÃO) ---
def main():
    known_encodings, known_names = load_known_faces()
    suspicious_plates = load_suspicious_plates()
    video_capture = cv2.VideoCapture(0)
    
    plate_stabilizer = TextStabilizer()

    # <<< MELHORIA: Variáveis para controle de frame skipping >>>

    frame_count = 0
    FRAME_SKIP = 5 # Process 1 out of every 5 frames
    last_faces = []
    last_plates = []
    plate_data = []  # Ensure plate_data is always defined

    if not video_capture.isOpened():
        print("Erro: Não foi possível abrir a câmera.")
        return


    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        frame_count += 1
        # Run heavy processing only on selected frames
        if frame_count % FRAME_SKIP == 0:
            last_faces, plate_data = process_frame(frame, known_encodings, known_names)
            # Update the stabilizer only with the most prominent plate result (the first)
            if plate_data:
                plate_stabilizer.update(plate_data[0]['text'])
            last_plates = plate_data if plate_data else []

        # Draw results on ALL frames for a smooth UI
        for face in last_faces:
            if face['name'] != "Desconhecido":
                trigger_alert(f"Rosto suspeito: {face['name']}", frame, face['box'])

        stable_plate_text = plate_stabilizer.get_stable_text()

        # Use the coordinates of the first detected plate to draw the stable text
        if last_plates:
            (x1, y1, x2, y2) = last_plates[0]['box']
            box_color = (255, 0, 0) # Blue
            if stable_plate_text and stable_plate_text in suspicious_plates:
                box_color = (0, 0, 255) # Red
                trigger_alert(f"Placa: {stable_plate_text}", frame, (x1, y1, x2, y2))

            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(frame, stable_plate_text, (x1, y1 - 15), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0,0,0), 5)
            cv2.putText(frame, stable_plate_text, (x1, y1 - 15), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,255,255), 2)

        cv2.imshow('Sistema de Vigilancia', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()