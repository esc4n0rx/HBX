import sys
import os
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QFileDialog, QListWidget, QProgressBar, 
                             QLabel, QTableWidget, QTableWidgetItem, QHeaderView)
from PyQt5.QtCore import QThread, pyqtSignal
from collections import Counter
from ultralytics import YOLO

# --- Thread de Trabalho para não congelar a interface ---
# A análise do YOLO pode demorar. Fazemos em uma thread separada.
class WorkerThread(QThread):
    # Sinais que serão enviados da thread para a janela principal
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    processing_finished = pyqtSignal(dict)

    def __init__(self, image_paths, model):
        super().__init__()
        self.image_paths = image_paths
        self.model = model

    def run(self):
        total_images = len(self.image_paths)
        total_counts = Counter()
        
        for i, img_path in enumerate(self.image_paths):
            self.status_updated.emit(f"Processando: {os.path.basename(img_path)}...")
            try:
                # Executa a predição
                results = self.model(img_path, verbose=False) # verbose=False para não poluir o console
                
                # Processa o resultado da imagem atual
                result = results[0]
                nomes_classes = result.names
                classes_detectadas_ids = result.boxes.cls.tolist()
                nomes_detectados = [nomes_classes[int(id)] for id in classes_detectadas_ids]
                
                # Acumula a contagem
                total_counts.update(nomes_detectados)

            except Exception as e:
                self.status_updated.emit(f"Erro ao processar {os.path.basename(img_path)}: {e}")

            # Atualiza a barra de progresso
            progress_percentage = int(((i + 1) / total_images) * 100)
            self.progress_updated.emit(progress_percentage)
            
        self.status_updated.emit("Processamento concluído!")
        self.processing_finished.emit(dict(total_counts))


# --- Janela Principal da Aplicação ---
class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Contador de Ativos com IA')
        self.setGeometry(100, 100, 800, 600)
        self.image_paths = []

        # Carrega o modelo YOLO
        try:
            self.model = YOLO('best.pt')
        except Exception as e:
            print(f"ERRO CRÍTICO: Não foi possível carregar 'best.pt'. {e}")
            sys.exit(1)

        # Layout principal
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Lista de imagens
        self.list_widget = QListWidget()
        self.layout.addWidget(self.list_widget)

        # Botões
        button_layout = QHBoxLayout()
        self.btn_load = QPushButton('Carregar Imagens')
        self.btn_load.clicked.connect(self.load_images)
        self.btn_process = QPushButton('Processar')
        self.btn_process.clicked.connect(self.start_processing)
        button_layout.addWidget(self.btn_load)
        button_layout.addWidget(self.btn_process)
        self.layout.addLayout(button_layout)

        # Status e Progresso
        self.status_label = QLabel('Pronto. Carregue as imagens para começar.')
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.layout.addWidget(self.status_label)
        self.layout.addWidget(self.progress_bar)

        # Tabela de resultados
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(2)
        self.results_table.setHorizontalHeaderLabels(['Item Detectado', 'Quantidade'])
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.layout.addWidget(self.results_table)

    def load_images(self):
        # Abre a janela para selecionar arquivos
        paths, _ = QFileDialog.getOpenFileNames(self, "Selecione uma ou mais imagens", "", "Imagens (*.png *.xpm *.jpg *.jpeg)")
        if paths:
            self.image_paths.extend(paths)
            self.list_widget.addItems([os.path.basename(p) for p in paths])
            self.status_label.setText(f"{len(self.image_paths)} imagem(s) carregada(s).")

    def start_processing(self):
        if not self.image_paths:
            self.status_label.setText("Nenhuma imagem carregada para processar.")
            return

        # Desabilita botões para evitar cliques duplos
        self.btn_process.setEnabled(False)
        self.btn_load.setEnabled(False)
        self.progress_bar.setValue(0)
        
        # Cria e inicia a thread de trabalho
        self.worker = WorkerThread(self.image_paths, self.model)
        self.worker.progress_updated.connect(self.progress_bar.setValue)
        self.worker.status_updated.connect(self.status_label.setText)
        self.worker.processing_finished.connect(self.display_results)
        self.worker.start()

    def display_results(self, counts):
        self.results_table.setRowCount(0) # Limpa a tabela
        if not counts:
            self.status_label.setText("Processamento concluído. Nenhum objeto detectado.")
        else:
            for i, (item, quantity) in enumerate(counts.items()):
                self.results_table.insertRow(i)
                self.results_table.setItem(i, 0, QTableWidgetItem(str(item)))
                self.results_table.setItem(i, 1, QTableWidgetItem(str(quantity)))
        
        # Habilita os botões novamente e limpa a lista para um novo lote
        self.btn_process.setEnabled(True)
        self.btn_load.setEnabled(True)
        self.image_paths = []
        self.list_widget.clear()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec_())