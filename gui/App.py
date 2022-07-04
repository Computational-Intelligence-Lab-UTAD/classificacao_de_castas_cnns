from PyQt5.QtCore import QSize, Qt
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QFrame, QComboBox, QLabel, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QFileDialog
from PyQt5.QtGui import QPixmap, QImage
import cv2
import sys
from Relatorio import Relatorio
import json

from model_manager import ModelManager
from explicador import Explicador

# Subclass QMainWindow to customize your application's main window
class MainWindow(QMainWindow):
    select_image_str = "Selecionar a imagem"
    model_str = "Selecione o modelo"
    xai_str = "Selecione o explicador"
    bt_image_str = "Selecionar"
    run_str = "Executar"

    MODEL_MANAGER = ModelManager()
    EXPLICADOR = Explicador()

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Classificação de Castas")
        self.setFixedSize(QSize(600, 480))

        #central widget
        self.central_widget = QWidget()               
        self.setCentralWidget(self.central_widget)
        central_layout = QHBoxLayout(self.central_widget)

        #principais layouts
        self.left_layout = QVBoxLayout()

        self.image_layout = QHBoxLayout()
        self.model_layout = QHBoxLayout()
        self.xai_layout = QHBoxLayout()

        self.run_layout = QGridLayout()

        #componentes iterativos
        self.select_image_button = QPushButton()
        self.select_image_button.setText(self.bt_image_str)

        self.select_model_combobox = QComboBox()
        self.select_xai_combobox = QComboBox()
        
        self.select_run = QPushButton()
        self.select_run.setText(self.run_str)
        self.select_run.setStyleSheet("background-color: green; color: white;")

        #componentes nao iterativos
        self.select_image_label = QLabel()
        self.select_image_label.setText(self.select_image_str)

        self.select_model_label = QLabel()
        self.select_model_label.setText(self.model_str)

        self.select_xai_label = QLabel()
        self.select_xai_label.setText(self.xai_str)

        #visualizador da imagem
        self.visualiza_imagem = QLabel()

        #relatorio
        self.relatorio_widget = Relatorio()
        
        #delete after
        self.relatorio_widget.configure('', 0)
        
        #povoa layouts
        self.image_layout.addWidget(self.select_image_label)
        self.image_layout.addWidget(self.select_image_button)

        self.model_layout.addWidget(self.select_model_label)
        self.model_layout.addWidget(self.select_model_combobox)

        self.xai_layout.addWidget(self.select_xai_label)
        self.xai_layout.addWidget(self.select_xai_combobox)

        self.run_layout.addWidget(self.select_run, 0, 3)

        self.left_layout.addLayout(self.image_layout)
        self.left_layout.addLayout(self.model_layout)
        self.left_layout.addLayout(self.xai_layout)
        self.left_layout.addLayout(self.run_layout)
        self.left_layout.addWidget(self.relatorio_widget)
        self.relatorio_widget.hide_labels()

        central_layout.addLayout(self.left_layout)
        central_layout.addWidget(self.visualiza_imagem)

        #carrega configuracoes 
        self.carrega_configuracoes()

        #define outras variáveis
        self.image = None

        #define iteractive actions actions
        self.define_actions()

        
    
    #metodo que seta imagem que vai ser visualizada
    def view_image(self, img):
        pixmap = QPixmap(img)
        pixmap = pixmap.scaled(300, 300)
        self.visualiza_imagem.setPixmap(pixmap)

    def carrega_configuracoes(self):
        self.path_config = "config.json"
        if len(sys.argv) > 1:
            self.path_config = sys.argv[1]
        
        f = open(self.path_config)
        self.configs = json.load(f)
        f.close()
        models = [x["name"] for x in self.configs["models"]]
        self.select_model_combobox.addItems(models)

        self.select_xai_combobox.addItems(self.EXPLICADOR.get_explainers_name())
        self.models = self.configs["models"]
        self.MODEL_MANAGER.configure_classes(self.configs["classes"])
    
    #define acoes dos botoes
    def define_actions(self):
        self.select_image_button.clicked.connect(self.open_image)
        self.select_run.clicked.connect(self.execute)

    #metodo que abre a janela para selecionar o arquivo
    def open_image(self):
        fname = QFileDialog.getOpenFileName(self, self.select_image_str,
            filter="Image files (*.jpg *.tiff *.jpeg *.bmp *.png)")
        print(fname)
        self.image = fname[0]
        self.view_image(self.image)
        self.relatorio_widget.hide_labels()

    #metodo que executa as coisas
    def execute(self):
        #obtem o modelo
        model = self.models[self.select_model_combobox.currentIndex()]
        model_info = self.MODEL_MANAGER.configure_model(model["name"], model["directory"], model["size"], model["type"])

        predicted = self.MODEL_MANAGER.make_prediction(model["name"], self.image)
        self.relatorio_widget.configure(predicted["class_name"], predicted["score"])
        #obtem a explicacao
        explainer = str(self.select_xai_combobox.currentText())
        if explainer == "Grad-CAM" or explainer == "Grad-CAM++":
            props = {
                "conv_layer":model["last_conv"]
            }
        else:
            props = {}
        
        explanation = self.EXPLICADOR.get_explanation(self.image, model_info["model"],  model_info["size"],  explainer,  model_info["preprocessing"], props=props)
        explanation = np.asarray(explanation)
        height, width, channel = explanation.shape
        cv2.imshow("", np.asarray(explanation))
        bytesPerLine = 3 * width
        explanation = QImage(explanation.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()

        self.view_image(explanation)
        

        






        

    


app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec()