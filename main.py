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
        """Adiciona um novo texto lido ao buffer se tiver o comprimento mínimo."""
        if new_text and len(new_text) >= self.min_len:
            self.buffer.append(new_text)

    def get_stable_text(self):
        """Retorna o texto mais comum no buffer se ele for predominante."""
        if not self.buffer:
            return self.stable_text # Retorna o último valor estável se o buffer estiver vazio
        
        counts = Counter(self.buffer)
        most_common_text, count = counts.most_common(1)[0]
        
        # Só atualiza se a leitura mais comum aparecer em pelo menos 30% do buffer
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
def carregar_rostos_conhecidos(path='./database/faces'):
    encodings_conhecidos, nomes_conhecidos = [], []
    print("Carregando base de dados de rostos...")
    for filename in os.listdir(path):
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            try:
                image_path = os.path.join(path, filename)
                imagem = face_recognition.load_image_file(image_path)
                encoding = face_recognition.face_encodings(imagem)[0]
                encodings_conhecidos.append(encoding)
                nomes_conhecidos.append(os.path.splitext(filename)[0])
            except (IndexError, Exception) as e:
                print(f"AVISO: Erro ao processar {filename}: {e}")
    print(f"{len(nomes_conhecidos)} rostos carregados.")
    return encodings_conhecidos, nomes_conhecidos

def carregar_placas_suspeitas(path='./database/placas_suspeitas.txt'):
    print("Carregando base de dados de placas...")
    try:
        with open(path, 'r') as f:
            placas = [limpar_texto_placa(line) for line in f]
        print(f"{len(placas)} placas carregadas.")
        return placas
    except FileNotFoundError:
        print(f"AVISO: Arquivo de placas '{path}' não encontrado.")
        return []

def emitir_alerta(mensagem, frame, box_coords=None):
    print(f"ALERTA: {mensagem}")
    if box_coords:
        (startX, startY, endX, endY) = box_coords
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 3)
        cv2.putText(frame, "ALERTA:", (box_coords[0], startY - 10),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 2)
    # Exibe a mensagem de alerta principal no topo da tela
    cv2.putText(frame, mensagem, (10, 30),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 255), 2)


def limpar_texto_placa(texto):
    return re.sub(r'[^A-Z0-9]', '', texto).upper()

# --- FUNÇÃO DE PROCESSAMENTO PRINCIPAL (ATUALIZADA) ---
def processar_frame(frame, encodings_conhecidos, nomes_conhecidos, placas_suspeitas, plate_stabilizer):
    # --- 1. RECONHECIMENTO FACIAL (sem alterações) ---
    rgb_small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)[:, :, ::-1]
    rgb_small_frame = np.ascontiguousarray(rgb_small_frame)
    locais_rostos = face_recognition.face_locations(rgb_small_frame)
    encodings_rostos = face_recognition.face_encodings(rgb_small_frame, locais_rostos)
    for face_encoding, face_location in zip(encodings_rostos, locais_rostos):
        matches = face_recognition.compare_faces(encodings_conhecidos, face_encoding, tolerance=0.5)
        nome = "Desconhecido"
        face_distances = face_recognition.face_distance(encodings_conhecidos, face_encoding)
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                nome = nomes_conhecidos[best_match_index]
                top, right, bottom, left = [c * 2 for c in face_location]
                emitir_alerta(f"Rosto suspeito: {nome}", frame, (left, top, right, bottom))

    # --- 2. PIPELINE DE RECONHECIMENTO DE PLACAS (ATUALIZADO) ---
    plate_results = yolo_plate_model(frame, verbose=False)
    
    current_frame_plates = []
    for result in plate_results:
        for box in result.boxes:
            x1, y1, x2, y2 = [int(i) for i in box.xyxy[0]]
            
            # a. Correção de Perspectiva
            src_pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype="float32")
            width, height = 400, 120
            dst_pts = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warped_plate = cv2.warpPerspective(frame, M, (width, height))
            gray = cv2.cvtColor(warped_plate, cv2.COLOR_BGR2GRAY)

            # b. Multi-Estratégia de Pré-processamento
            # Estratégia A: Contraste
            sharpened = cv2.addWeighted(gray, 2.5, cv2.GaussianBlur(gray, (0, 0), 3.0), -1.5, 0)
            _, ocr_image_A = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            if np.mean(ocr_image_A) < 128: ocr_image_A = cv2.bitwise_not(ocr_image_A)

            # Estratégia B: Iluminação
            denoised_bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
            ocr_image_B = cv2.adaptiveThreshold(denoised_bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            if np.mean(ocr_image_B) > 128: ocr_image_B = cv2.bitwise_not(ocr_image_B)

            # c. Leitura e Seleção
            try:
                custom_config = r'--oem 3 --psm 7 -l por -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                placa_A = limpar_texto_placa(pytesseract.image_to_string(ocr_image_A, config=custom_config))
                placa_B = limpar_texto_placa(pytesseract.image_to_string(ocr_image_B, config=custom_config))

                comprimento_ideal = 7
                len_A, len_B = len(placa_A), len(placa_B)
                
                placa_limpa = ""
                if len_A == comprimento_ideal and len_B != comprimento_ideal:
                    placa_limpa = placa_A
                elif len_B == comprimento_ideal and len_A != comprimento_ideal:
                    placa_limpa = placa_B
                else:
                    placa_limpa = placa_A if len_A >= len_B else placa_B
                
                # d. Atualizar o Estabilizador com o melhor resultado do frame
                plate_stabilizer.update(placa_limpa)
                current_frame_plates.append({'box': (x1, y1, x2, y2)})
                
            except Exception:
                continue

    # e. Desenhar o resultado estável na tela
    stable_plate_text = plate_stabilizer.get_stable_text()
    for plate_info in current_frame_plates:
        (x1, y1, x2, y2) = plate_info['box']
        cor_caixa = (255, 0, 0) # Azul
        
        if stable_plate_text and stable_plate_text in placas_suspeitas:
            cor_caixa = (0, 0, 255) # Vermelho
            emitir_alerta(f"Placa: {stable_plate_text}", frame, (x1, y1, x2, y2))
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), cor_caixa, 2)
        cv2.putText(frame, stable_plate_text, (x1, y1 - 15), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 0), 5)
        cv2.putText(frame, stable_plate_text, (x1, y1 - 15), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)

# --- FUNÇÃO MAIN (sem alterações) ---
def main():
    encodings_conhecidos, nomes_conhecidos = carregar_rostos_conhecidos()
    placas_suspeitas = carregar_placas_suspeitas()
    video_capture = cv2.VideoCapture(0)
    
    plate_stabilizer = TextStabilizer()

    if not video_capture.isOpened():
        print("Erro: Não foi possível abrir a câmera.")
        return

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        processar_frame(frame, encodings_conhecidos, nomes_conhecidos, placas_suspeitas, plate_stabilizer)
        cv2.imshow('Sistema de Vigilancia', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()