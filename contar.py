import sys
from ultralytics import YOLO
from collections import Counter

# --- VALIDAÇÃO INICIAL ---
# Verifica se o caminho da imagem foi fornecido como argumento
if len(sys.argv) < 2:
    print("ERRO: Forneça o caminho da imagem como argumento.")
    print("Exemplo: python contar.py imagens_para_contar/pallet1.jpg")
    sys.exit(1) # Encerra o script com um código de erro

# Pega o caminho da imagem a partir do argumento da linha de comando
caminho_imagem = sys.argv[1]

# --- CARREGAMENTO DO MODELO ---
try:
    # Carrega o seu modelo treinado. Certifique-se que o arquivo 'best.pt'
    # está na mesma pasta que este script.
    model = YOLO('best.pt')
except Exception as e:
    print(f"ERRO: Não foi possível carregar o modelo 'best.pt'. Verifique se o arquivo está no lugar certo.")
    print(e)
    sys.exit(1)

# --- EXECUÇÃO DA PREDIÇÃO ---
try:
    # Executa a predição na imagem fornecida
    results = model(caminho_imagem)
except Exception as e:
    print(f"ERRO: Falha ao processar a imagem: {caminho_imagem}")
    print(e)
    sys.exit(1)


# --- PROCESSAMENTO E CONTAGEM DOS RESULTADOS ---

# O resultado 'results' é uma lista. Como processamos uma imagem, pegamos o primeiro item.
result = results[0]

# Pega os nomes das classes que o modelo conhece (ex: 'hb_618', 'hb_623')
nomes_classes = result.names

# Pega as classes de cada caixa detectada (em formato de número, ex: 0, 1)
classes_detectadas_ids = result.boxes.cls.tolist()

# Converte os IDs das classes para os nomes reais
nomes_detectados = [nomes_classes[int(id)] for id in classes_detectadas_ids]

# Usa a biblioteca Counter para contar quantas vezes cada nome aparece
contagem = Counter(nomes_detectados)

# --- EXIBIÇÃO DO RESULTADO ---
print("\n--- CONTAGEM DE CAIXAS ---")
if not contagem:
    print("Nenhuma caixa foi detectada na imagem.")
else:
    for nome_caixa, quantidade in contagem.items():
        print(f"{nome_caixa}: {quantidade} unidade(s)")
print("--------------------------\n")

# --- SALVAR IMAGEM COM DETECÇÕES (OPCIONAL) ---
# Salva a imagem com as caixas desenhadas para você poder verificar
nome_arquivo_resultado = "resultado_contagem.jpg"
result.save(filename=nome_arquivo_resultado)
print(f"Imagem com as detecções foi salva como '{nome_arquivo_resultado}'")