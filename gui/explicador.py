import tensorflow as tf
import tensorflow.keras as keras
from gradcam_explainer import GradCAMExplainer
from gradcampp_explainer import GradCAMPPExplainer

class Explicador:
    explainers = {"Lime": None,
    "Grad-CAM": GradCAMExplainer(), 
    "Grad-CAM++":GradCAMPPExplainer()}
    
    #retorna nome dos explicadores
    def get_explainers_name(self):
        return list(self.explainers.keys()).copy()

    #metodo utilizado para retornar uma explicacao
    def get_explanation(self, img_path, model, size, explainer, prepocess_input, index=None, props=None):
        explainer = self.explainers[explainer]
        return explainer.get_explanation(img_path, model, size, props, prepocess_input, index)
        


