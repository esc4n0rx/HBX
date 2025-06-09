import os
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from functools import wraps
from werkzeug.exceptions import RequestEntityTooLarge

from config import Config
from models.analyzer import BoxAnalyzer
from utils.validators import validate_image, convert_to_numpy

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Inicializar Flask
app = Flask(__name__)
app.config.from_object(Config)

# Configurar CORS
CORS(app, origins=["*"])  # Configure conforme necessário para produção

# Inicializar analisador
try:
    analyzer = BoxAnalyzer()
    logger.info("✅ Analisador inicializado com sucesso")
except Exception as e:
    logger.error(f"❌ Erro ao inicializar analisador: {e}")
    analyzer = None

def require_auth(f):
    """Decorator para autenticação via token"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Verifica token no header Authorization
        auth_header = request.headers.get('Authorization')
        if not auth_header:
            return jsonify({
                "success": False,
                "error": "Token de autorização necessário",
                "message": "Inclua o header: Authorization: Bearer <seu_token>"
            }), 401
        
        # Extrai o token
        try:
            token = auth_header.split(' ')[1] if auth_header.startswith('Bearer ') else auth_header
        except IndexError:
            return jsonify({
                "success": False,
                "error": "Formato de token inválido",
                "message": "Use: Authorization: Bearer <seu_token>"
            }), 401
        
        if token != Config.API_TOKEN:
            return jsonify({
                "success": False,
                "error": "Token inválido"
            }), 401
        
        return f(*args, **kwargs)
    
    return decorated_function

@app.errorhandler(413)
@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    return jsonify({
        "success": False,
        "error": "Arquivo muito grande",
        "message": f"Tamanho máximo permitido: {Config.MAX_CONTENT_LENGTH // (1024*1024)}MB"
    }), 413

@app.errorhandler(Exception)
def handle_general_error(e):
    logger.error(f"Erro não tratado: {e}")
    return jsonify({
        "success": False,
        "error": "Erro interno do servidor",
        "message": "Tente novamente mais tarde"
    }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint de verificação de saúde"""
    return jsonify({
        "success": True,
        "status": "healthy",
        "analyzer_ready": analyzer is not None
    })

@app.route('/analyze', methods=['POST'])
@require_auth
def analyze_image():
    """
    Endpoint principal para análise de imagens
    
    Espera:
    - Header: Authorization: Bearer <token>
    - Form-data: file (imagem)
    
    Retorna:
    - JSON com contagens das caixas
    """
    
    # Verifica se o analisador está disponível
    if analyzer is None:
        return jsonify({
            "success": False,
            "error": "Serviço indisponível",
            "message": "Modelos de IA não puderam ser carregados"
        }), 503
    
    # Verifica se foi enviado um arquivo
    if 'file' not in request.files:
        return jsonify({
            "success": False,
            "error": "Nenhum arquivo fornecido",
            "message": "Envie uma imagem no campo 'file'"
        }), 400
    
    file = request.files['file']
    
    # Valida o arquivo
    is_valid, message = validate_image(file)
    if not is_valid:
        return jsonify({
            "success": False,
            "error": "Arquivo inválido",
            "message": message
        }), 400
    
    try:
        # Converte para numpy array
        img_array, error = convert_to_numpy(file)
        if error:
            return jsonify({
                "success": False,
                "error": "Erro no processamento",
                "message": error
            }), 400
        
        # Analisa a imagem
        result = analyzer.analyze_image(img_array)
        
        logger.info(f"Análise concluída: {result['data']['summary']['total_processed']} caixas processadas")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Erro na análise: {e}")
        return jsonify({
            "success": False,
            "error": "Erro na análise",
            "message": "Não foi possível processar a imagem"
        }), 500

if __name__ == '__main__':
    if not analyzer:
        logger.error("❌ Não é possível iniciar sem os modelos. Verifique os arquivos etiqueta.pt e best.pt")
        exit(1)
    
    logger.info(f"🚀 Iniciando servidor na porta {Config.PORT}")
    app.run(
        host=Config.HOST,
        port=Config.PORT,
        debug=Config.DEBUG
    )