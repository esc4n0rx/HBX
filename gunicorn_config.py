import os
import multiprocessing

# Configurações básicas
bind = f"0.0.0.0:{os.getenv('PORT', '7000')}"
workers = min(4, multiprocessing.cpu_count())
worker_class = "sync"
worker_connections = 1000

# Timeouts (importantes para processamento de imagem)
timeout = 300  # 5 minutos para processamento de imagem
keepalive = 2
max_requests = 500
max_requests_jitter = 50

# Performance
preload_app = True
worker_tmp_dir = "/dev/shm"

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"
capture_output = True
enable_stdio_inheritance = True

# Security
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190

# Process naming
proc_name = "box-analyzer-api"

# Graceful shutdown
graceful_timeout = 30
