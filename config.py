import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    API_TOKEN = os.getenv('API_TOKEN')
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', 7000))
    
    # Validações de segurança
    if not API_TOKEN:
        raise ValueError("API_TOKEN deve ser definido no arquivo .env")
    
    if len(API_TOKEN) < 20:
        raise ValueError("API_TOKEN deve ter pelo menos 20 caracteres")

    # Configurações de upload
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB máximo
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}