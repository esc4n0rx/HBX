"""
BACKEND DE ANÁLISE DE ATIVOS v1.0 - com Flask

Este servidor expõe uma API na rota /analisar que aceita o envio de uma imagem
e retorna a contagem de ativos em formato JSON.

COMO USAR:
1. Execute este script: python server.py
2. Envie uma requisição POST para http://127.0.0.1:5000/analisar
   com a imagem no campo 'imagem' de um formulário multipart/form-data.
"""
import cv2
import numpy as np
from flask import Flask, request, jsonify

# Importa as bibliotecas de IA
from ultralytics import YOLO
from pyzbar.pyzbar import decode
import zxingcpp
import easyocr

print("==============================================")
print("  INICIALIZANDO SERVIDOR DE ANÁLISE DE ATIVOS   ")
print("==============================================")

# --- CARREGAMENTO PESADO FEITO APENAS UMA VEZ ---
# Carregamos os modelos na memória quando o servidor inicia.
# Isso evita recarregá-los a cada requisição, tornando a API muito mais rápida.
try:
    MODELO_YOLO = YOLO('etiqueta.pt')
    LEITOR_OCR = easyocr.Reader(['en'], gpu=False)
    print("✅ Modelos YOLO e OCR carregados e prontos.")
except Exception as e:
    print(f"❌ ERRO CRÍTICO AO CARREGAR MODELOS: {e}")
    # Se os modelos não carregam, o servidor não pode funcionar.
    exit()

# --- INICIALIZAÇÃO DO FLASK ---
app = Flask(__name__)

# --- ROTA DA API ---
@app.route('/analisar', methods=['POST'])
def analisar_imagem():
    """
    Esta função é chamada quando uma requisição POST é feita para /analisar.
    """
    print("\n▶️  Recebida nova requisição em /analisar...")

    # Verifica se a imagem foi enviada corretamente
    if 'imagem' not in request.files:
        print("  ❌ Erro: Nenhuma imagem encontrada na requisição.")
        return jsonify({"erro": "Nenhum arquivo de imagem enviado"}), 400

    file = request.files['imagem']
    if file.filename == '':
        print("  ❌ Erro: Arquivo de imagem inválido.")
        return jsonify({"erro": "Nenhum arquivo selecionado"}), 400

    try:
        # Lê a imagem enviada e a converte para um formato que o OpenCV entende
        filestr = file.read()
        npimg = np.frombuffer(filestr, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"  ❌ Erro ao decodificar a imagem: {e}")
        return jsonify({"erro": f"Não foi possível ler o arquivo de imagem: {e}"}), 400

    # --- LÓGICA DE ANÁLISE (adaptada do script v3) ---
    # Para esta requisição específica, inicializamos os contadores.
    # Note que os sets não funcionam para contagem única entre MÚLTIPLAS requisições,
    # a menos que você adicione um banco de dados (passo futuro).
    counted_barcodes_618, counted_barcodes_623 = set(), set()
    visual_detections_618, visual_detections_623, etiquetas_nao_identificadas = 0, 0, 0

    results = MODELO_YOLO(img, verbose=False)
    etiquetas_detectadas = results[0].boxes

    for etiqueta in etiquetas_detectadas:
        x1, y1, x2, y2 = map(int, etiqueta.xyxy[0])
        label_crop = img[y1:y2, x1:x2]
        
        identificado = False
        id_unico = None

        try:
            gray_crop = cv2.cvtColor(label_crop, cv2.COLOR_BGR2GRAY)
            barcodes_pyzbar = decode(gray_crop)
            if barcodes_pyzbar: id_unico = barcodes_pyzbar[0].data.decode('utf-8')
            else:
                barcodes_zxing = zxingcpp.read_barcodes(gray_crop)
                if barcodes_zxing: id_unico = barcodes_zxing[0].text

            if id_unico:
                if "5592261800" in id_unico:
                    counted_barcodes_618.add(id_unico)
                    identificado = True
                elif "5592262300" in id_unico:
                    counted_barcodes_623.add(id_unico)
                    identificado = True
        except: pass

        if identificado: continue

        try:
            ocr_result = LEITOR_OCR.readtext(label_crop, detail=0, paragraph=False)
            text_detectado = "".join(ocr_result).lower().replace(" ", "")
            if "618" in text_detectado:
                visual_detections_618 += 1
                identificado = True
            elif "623" in text_detectado:
                visual_detections_623 += 1
                identificado = True
        except: pass
        
        if not identificado:
            etiquetas_nao_identificadas += 1

    # --- MONTAGEM DA RESPOSTA JSON ---
    resultado_final = {
        "status": "sucesso",
        "contagem_confirmada": {
            "caixas_618": len(counted_barcodes_618),
            "caixas_623": len(counted_barcodes_623)
        },
        "deteccoes_visuais": {
            "caixas_618": visual_detections_618,
            "caixas_623": visual_detections_623
        },
        "etiquetas_nao_identificadas": etiquetas_nao_identificadas
    }
    
    print(f"  ✅ Análise concluída. Respondendo com JSON.")
    return jsonify(resultado_final)

# --- INICIA O SERVIDOR ---
if __name__ == '__main__':
    # host='0.0.0.0' faz o servidor ser acessível por outros dispositivos na sua rede
    app.run(host='0.0.0.0', port=5000, debug=True)