import os
from roboflow import Roboflow
from ultralytics import YOLO
from dotenv import load_dotenv

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

# --- CONFIGURAÇÃO ---
# ATENÇÃO: Cole sua chave de API privada do Roboflow aqui.
# Mantenha esta chave em segredo!
ROBOFLOW_API_KEY = os.getenv('ROBOFLOW_API_KEY')

# Parâmetros do Treinamento
EPOCHS = 50 # Quantas épocas você quer treinar
IMG_SIZE = 640 # Tamanho da imagem

# --- DOWNLOAD DO DATASET ---
print("➡️ Iniciando o download do dataset do Roboflow...")

try:
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace("cd-af9ha").project("hbx-vzhod")
    version = project.version(1)
    dataset = version.download("yolov8")
    
    # Pega o caminho do arquivo .yaml, que é necessário para o treino
    dataset_yaml_path = os.path.join(dataset.location, 'data.yaml')
    print("✅ Download do dataset concluído com sucesso!")
    print(f"Dataset salvo em: {dataset.location}")

except Exception as e:
    print(f"❌ ERRO: Falha ao baixar o dataset do Roboflow.")
    print(f"   Verifique sua chave de API e conexão com a internet.")
    print(f"   Detalhe do erro: {e}")
    exit() # Encerra o script se o download falhar


# --- TREINAMENTO DO MODELO ---
print("\n➡️ Iniciando o treinamento do modelo YOLOv8...")
print(f"   Treinando por {EPOCHS} épocas com imagens de tamanho {IMG_SIZE}.")
print("   Aguarde... Isso pode levar um bom tempo dependendo do seu hardware.")

try:
    # Carrega um modelo pré-treinado (yolov8n.pt é o menor e mais rápido)
    model = YOLO('yolov8n.pt')

    # Inicia o treinamento
    # O YOLO detectará automaticamente se você tem uma GPU com CUDA e a usará.
    # Caso contrário, usará a CPU (muito mais lento).
    results = model.train(
        data=dataset_yaml_path,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        # Você pode adicionar mais argumentos aqui se desejar
        # project="runs/train", # Opcional: define a pasta de resultados
        # name="experimento_local" # Opcional: define o nome do subdiretório
    )
    
    print("\n✅ Treinamento concluído com sucesso!")
    # O modelo final (best.pt) será salvo na pasta 'runs/detect/train/'

except Exception as e:
    print(f"❌ ERRO: Ocorreu um problema durante o treinamento.")
    print(f"   Detalhe do erro: {e}")