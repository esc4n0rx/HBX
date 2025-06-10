# Use uma imagem base Python oficial.
# É importante usar uma versão que seja compatível com suas dependências (especialmente PyTorch/Ultralytics)
# e que inclua as bibliotecas necessárias para o processamento de imagem.
# Optamos por uma imagem com "slim-buster" ou "slim-bullseye" para reduzir o tamanho da imagem final.
FROM python:3.10-slim-bullseye

# Define o diretório de trabalho dentro do contêiner
WORKDIR /app

# Instala as dependências do sistema operacional necessárias para OpenCV, zbar e outras bibliotecas de imagem.
# apt-get update e clean são para garantir que as instalações sejam eficientes e que a imagem final seja menor.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libfontconfig1 \
    libice6 \
    libgomp1 \
    libgthread-2.0-0 \
    libzbar0 \
    libzbar-dev \
    && rm -rf /var/lib/apt/lists/*

# Copia o arquivo de requisitos para o diretório de trabalho antes de instalar as dependências Python.
# Isso otimiza o cache do Docker: se requirements.txt não mudar, essa camada não será reconstruída.
COPY requirements.txt .

# Instala as dependências Python.
# O parâmetro --no-cache-dir é para evitar o cache de pacotes, reduzindo o tamanho da imagem.
RUN pip install --no-cache-dir -r requirements.txt

# Copia o restante do código da aplicação para o contêiner.
# Copiar o diretório de modelos explicitamente garante que os arquivos .pt estejam presentes.
COPY . /app

# Define variáveis de ambiente para a aplicação.
# Coolify pode sobrescrever estas, mas é bom ter padrões.
ENV HOST=0.0.0.0
ENV PORT=7000
ENV API_TOKEN=HBX202500000000011100010101010101011010101010101

# Expõe a porta em que a aplicação será executada.
EXPOSE 7000

# Comando para iniciar a aplicação.
# O gunicorn é o servidor de produção que você já configurou em gunicorn_config.py.
CMD ["gunicorn", "-c", "gunicorn_config.py", "app:app"]