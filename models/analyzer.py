import cv2
import numpy as np
from ultralytics import YOLO
from pyzbar.pyzbar import decode
import zxingcpp
import easyocr
from PIL import Image
import logging
import os

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BoxAnalyzer:
    def __init__(self):
        """Inicializa os modelos de IA"""
        self.label_detector = None # Modelo para achar ETIQUETAS
        self.box_detector = None   # Modelo para achar CAIXAS INTEIRAS
        self.ocr_reader = None
        self._load_models()
    
    def _load_models(self):
        """Carrega todos os modelos necessários"""
        try:
            logger.info("Carregando modelos de IA...")
            
            # Renomeamos para clareza
            label_model_path = 'etiqueta.pt'
            box_model_path = 'best.pt'
            
            if not os.path.exists(label_model_path):
                raise FileNotFoundError(f"Arquivo '{label_model_path}' não encontrado")
            
            if not os.path.exists(box_model_path):
                raise FileNotFoundError(f"Arquivo '{box_model_path}' não encontrado")
            
            # Carrega os modelos
            self.label_detector = YOLO(label_model_path)
            self.box_detector = YOLO(box_model_path)
            self.ocr_reader = easyocr.Reader(['en'], gpu=False)
            
            logger.info("✅ Modelos carregados com sucesso!")
            
        except Exception as e:
            logger.error(f"❌ Erro ao carregar modelos: {e}")
            raise

    def analyze_image(self, image_array):
        """
        Analisa a imagem com um fluxo de duas etapas para maior precisão.
        """
        try:
            img = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            
            counted_barcodes_618 = 0
            counted_barcodes_623 = 0
            visual_detections_618 = 0
            visual_detections_623 = 0
            
            confirmed_boxes_coords = []

            # --- ETAPA 1: DETECÇÃO POR ETIQUETA / BARCODE / OCR ---
            logger.info("Etapa 1: Procurando por etiquetas...")
            label_results = self.label_detector(img, verbose=False)
            
            if label_results and label_results[0].boxes:
                for label_box in label_results[0].boxes:
                    x1, y1, x2, y2 = map(int, label_box.xyxy[0])
                    crop_label = img[y1:y2, x1:x2]
                    
                    if crop_label.size == 0: continue

                    # Tenta ler barcode
                    barcode_type = self._read_barcode(crop_label)
                    if barcode_type:
                        if barcode_type == "618_barcode": counted_barcodes_618 += 1
                        elif barcode_type == "623_barcode": counted_barcodes_623 += 1
                        confirmed_boxes_coords.append(label_box.xyxy[0].cpu().numpy())
                        continue # Pula para a próxima etiqueta

                    # Se não leu barcode, tenta OCR
                    ocr_type = self._read_text_ocr(crop_label)
                    if ocr_type:
                        if ocr_type == "618_visual": visual_detections_618 += 1
                        elif ocr_type == "623_visual": visual_detections_623 += 1
                        confirmed_boxes_coords.append(label_box.xyxy[0].cpu().numpy())

            # --- ETAPA 2: DETECÇÃO VISUAL DA CAIXA INTEIRA (com o novo modelo) ---
            logger.info("Etapa 2: Procurando por caixas inteiras...")
            box_results = self.box_detector(img, verbose=False)
            
            total_boxes_detected = 0
            if box_results and box_results[0].boxes:
                total_boxes_detected = len(box_results[0].boxes)
                for box in box_results[0].boxes:
                    # Verifica se esta caixa já foi contada na Etapa 1
                    is_already_counted = False
                    current_box_coords = box.xyxy[0].cpu().numpy()
                    for confirmed_coords in confirmed_boxes_coords:
                        # Se a sobreposição for alta, considera que já foi contada
                        if self._calculate_iou(current_box_coords, confirmed_coords) > 0.5:
                            is_already_counted = True
                            break
                    
                    # Se não foi contada, adiciona como uma detecção visual
                    if not is_already_counted:
                        class_name = self.box_detector.names[int(box.cls[0])]
                        if '618' in class_name:
                            visual_detections_618 += 1
                        elif '623' in class_name:
                            visual_detections_623 += 1

            return self._create_result(
                counted_barcodes_618,
                counted_barcodes_623,
                visual_detections_618,
                visual_detections_623,
                total_boxes_detected
            )
            
        except Exception as e:
            logger.error(f"Erro na análise da imagem: {e}")
            raise

    def _calculate_iou(self, boxA, boxB):
        """Calcula a Intersecção sobre União (IoU) entre duas caixas."""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def _read_barcode(self, crop_box):
        """Tenta ler código de barras da caixa"""
        try:
            gray_crop = cv2.cvtColor(crop_box, cv2.COLOR_BGR2GRAY)
            
            # Tenta com pyzbar
            barcodes_pyzbar = decode(gray_crop)
            if barcodes_pyzbar:
                barcode_data = barcodes_pyzbar[0].data.decode('utf-8', 'ignore')
                if "5592261800" in barcode_data: return "618_barcode"
                elif "5592262300" in barcode_data: return "623_barcode"
            
            # Tenta com zxingcpp se pyzbar falhar
            barcodes_zxing = zxingcpp.read_barcodes(Image.fromarray(gray_crop))
            if barcodes_zxing:
                barcode_data = barcodes_zxing[0].text
                if "5592261800" in barcode_data: return "618_barcode"
                elif "5592262300" in barcode_data: return "623_barcode"
        except Exception:
            pass
        return None
    
    def _read_text_ocr(self, crop_box):
        """Tenta ler texto usando OCR"""
        try:
            ocr_result = self.ocr_reader.readtext(crop_box, detail=0, paragraph=False)
            if ocr_result:
                text_detected = "".join(ocr_result).lower().replace(" ", "")
                if "618" in text_detected: return "618_visual"
                elif "623" in text_detected: return "623_visual"
        except Exception:
            pass
        return None
    
    # Esta função não é mais necessária, pois a classificação é feita na Etapa 2
    # def _classify_with_ai(self, crop_box): ...
    
    def _create_result(self, barcode_618, barcode_623, visual_618, visual_623, total):
        """Cria o resultado formatado"""
        return {
            "success": True,
            "data": {
                "confirmed_count": {
                    "boxes_618": barcode_618,
                    "boxes_623": barcode_623,
                    "total": barcode_618 + barcode_623
                },
                "visual_count": {
                    "boxes_618": visual_618,
                    "boxes_623": visual_623,
                    "total": visual_618 + visual_623
                },
                "summary": {
                    "total_boxes_detected_by_visual_model": total,
                    "total_processed": barcode_618 + barcode_623 + visual_618 + visual_623,
                    "boxes_618_total": barcode_618 + visual_618,
                    "boxes_623_total": barcode_623 + visual_623
                }
            }
        }