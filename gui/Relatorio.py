from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QWidget, QLabel, QHBoxLayout, QVBoxLayout


class Relatorio(QWidget):

    #constantes
    class_name_str = "Classe: {}"
    percent_str = "Percentual da classificação: {}%"

    def __init__(self):
        super().__init__()

        self.class_name_label = QLabel()
        self.percent_label = QLabel()


        central_layout = QVBoxLayout()
        central_layout.addWidget(self.class_name_label)
        central_layout.addWidget(self.percent_label)

        self.setLayout(central_layout)
        self.resize(self.width(), 100)

    def configure(self, name, percent):
        percent = round(float(percent)*100, 2)
        self.class_name_label.setText(self.class_name_str.format(name))
        self.percent_label.setText(self.percent_str.format(percent))
        self.class_name_label.show()
        self.percent_label.show()

    def hide_labels(self):
        self.class_name_label.hide()
        self.percent_label.hide()



