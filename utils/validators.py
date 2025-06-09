import os
from PIL import Image
import numpy as np
from werkzeug.utils import secure_filename
from config import Config

def allowed_file(filename):
    """Verifica se o arquivo tem uma extensão permitida"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

def validate_image(file):
    """Valida se o arquivo é uma imagem válida"""
    try:
        # Verifica o nome do arquivo
        if not file or not file.filename:
            return False, "Nenhum arquivo fornecido"
        
        if not allowed_file(file.filename):
            return False, f"Tipo de arquivo não permitido. Use: {', '.join(Config.ALLOWED_EXTENSIONS)}"
        
        # Tenta abrir a imagem
        image = Image.open(file.stream)
        image.verify()  # Verifica se é uma imagem válida
        
        # Reset do stream após verificação
        file.stream.seek(0)
        
        return True, "Imagem válida"
    
    except Exception as e:
        return False, f"Arquivo não é uma imagem válida: {str(e)}"

def convert_to_numpy(file):
    """Converte arquivo de imagem para array numpy"""
    try:
        image = Image.open(file.stream)
        # Converte para RGB se necessário
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Converte para numpy array
        img_array = np.array(image)
        return img_array, None
    
    except Exception as e:
        return None, f"Erro ao processar imagem: {str(e)}"