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
        self.detector_model = None
        self.classifier_model = None
        self.ocr_reader = None
        self._load_models()
    
    def _load_models(self):
        """Carrega todos os modelos necessários"""
        try:
            logger.info("Carregando modelos de IA...")
            
            # Verifica se os arquivos de modelo existem
            if not os.path.exists('etiqueta.pt'):
                raise FileNotFoundError("Arquivo 'etiqueta.pt' não encontrado")
            
            if not os.path.exists('best.pt'):
                raise FileNotFoundError("Arquivo 'best.pt' não encontrado")
            
            # Carrega os modelos
            self.detector_model = YOLO('etiqueta.pt')
            self.classifier_model = YOLO('best.pt')
            self.ocr_reader = easyocr.Reader(['en'], gpu=False)
            
            logger.info("✅ Modelos carregados com sucesso!")
            
        except Exception as e:
            logger.error(f"❌ Erro ao carregar modelos: {e}")
            raise
    
    def analyze_image(self, image_array):
        """
        Analisa a imagem e retorna contagens das caixas
        
        Args:
            image_array: numpy array da imagem
            
        Returns:
            dict: Resultado da análise com contagens
        """
        try:
            # Converte para formato BGR (OpenCV)
            img = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            
            # Inicializa contadores
            counted_barcodes_618 = set()
            counted_barcodes_623 = set()
            visual_detections_618 = 0
            visual_detections_623 = 0
            total_boxes = 0
            
            # Detecta caixas na imagem
            results_detector = self.detector_model(img, verbose=False)
            
            if not results_detector or len(results_detector) == 0:
                return self._create_result(0, 0, 0, 0, 0)
            
            boxes = results_detector[0].boxes
            if boxes is None:
                return self._create_result(0, 0, 0, 0, 0)
            
            # Processa cada caixa detectada
            for box in boxes:
                total_boxes += 1
                
                # Extrai coordenadas da caixa
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Recorta a região da caixa
                crop_box = img[y1:y2, x1:x2]
                
                if crop_box.size == 0:
                    logger.warning("Caixa vazia detectada, pulando...")
                    continue
                
                # Tenta identificar o tipo da caixa
                box_type = self._identify_box_type(crop_box)
                
                # Atualiza contadores baseado no resultado
                if box_type == "618_barcode":
                    counted_barcodes_618.add(f"barcode_{total_boxes}")
                elif box_type == "623_barcode":
                    counted_barcodes_623.add(f"barcode_{total_boxes}")
                elif box_type == "618_visual":
                    visual_detections_618 += 1
                elif box_type == "623_visual":
                    visual_detections_623 += 1
            
            return self._create_result(
                len(counted_barcodes_618),
                len(counted_barcodes_623),
                visual_detections_618,
                visual_detections_623,
                total_boxes
            )
            
        except Exception as e:
            logger.error(f"Erro na análise da imagem: {e}")
            raise
    
    def _identify_box_type(self, crop_box):
        """
        Identifica o tipo da caixa usando múltiplas técnicas
        
        Args:
            crop_box: Imagem recortada da caixa
            
        Returns:
            str: Tipo da caixa identificada
        """
        # Tentativa 1: Leitura de código de barras
        barcode_result = self._read_barcode(crop_box)
        if barcode_result:
            return barcode_result
        
        # Tentativa 2: OCR
        ocr_result = self._read_text_ocr(crop_box)
        if ocr_result:
            return ocr_result
        
        # Tentativa 3: Classificação por IA
        ai_result = self._classify_with_ai(crop_box)
        if ai_result:
            return ai_result
        
        return "unknown"
    
    def _read_barcode(self, crop_box):
        """Tenta ler código de barras da caixa"""
        try:
            gray_crop = cv2.cvtColor(crop_box, cv2.COLOR_BGR2GRAY)
            
            # Tenta com pyzbar
            barcodes_pyzbar = decode(gray_crop)
            if barcodes_pyzbar:
                barcode_data = barcodes_pyzbar[0].data.decode('utf-8', 'ignore')
                if "5592261800" in barcode_data:
                    return "618_barcode"
                elif "5592262300" in barcode_data:
                    return "623_barcode"
            
            # Tenta com zxingcpp se pyzbar falhar
            barcodes_zxing = zxingcpp.read_barcodes(Image.fromarray(gray_crop))
            if barcodes_zxing:
                barcode_data = barcodes_zxing[0].text
                if "5592261800" in barcode_data:
                    return "618_barcode"
                elif "5592262300" in barcode_data:
                    return "623_barcode"
                    
        except Exception as e:
            logger.debug(f"Erro na leitura de barcode: {e}")
        
        return None
    
    def _read_text_ocr(self, crop_box):
        """Tenta ler texto usando OCR"""
        try:
            ocr_result = self.ocr_reader.readtext(crop_box, detail=0, paragraph=False)
            if ocr_result:
                text_detected = "".join(ocr_result).lower().replace(" ", "")
                if "618" in text_detected:
                    return "618_visual"
                elif "623" in text_detected:
                    return "623_visual"
                    
        except Exception as e:
            logger.debug(f"Erro no OCR: {e}")
        
        return None
    
    def _classify_with_ai(self, crop_box):
        """Classifica usando modelo de IA"""
        try:
            class_results = self.classifier_model(crop_box, verbose=False)
            if class_results[0].probs is not None:
                pred_index = class_results[0].probs.top1
                class_name = class_results[0].names[pred_index]
                
                if '618' in str(class_name):
                    return "618_visual"
                elif '623' in str(class_name):
                    return "623_visual"
                    
        except Exception as e:
            logger.debug(f"Erro na classificação IA: {e}")
        
        return None
    
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
                    "total_boxes_detected": total,
                    "total_processed": barcode_618 + barcode_623 + visual_618 + visual_623,
                    "boxes_618_total": barcode_618 + visual_618,
                    "boxes_623_total": barcode_623 + visual_623
                }
            }
        }