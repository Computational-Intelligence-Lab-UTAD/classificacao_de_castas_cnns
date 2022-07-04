import tensorflow as tf
import tensorflow.keras as keras
import util
import numpy as np

class ModelManager:
    models = {}
    TYPE_XCEPTION = "xception"
    TYPE_RESNET101 = "resenet101"
    TYPE_EFFICIENTNETB3 = "efficientnetb3"


    #adicona um modelo à coleção carregando ele no keras
    def _add_model(self, name, path, size, type_model):
        container = {}
        
        #importa modelo
        container["model"] = tf.keras.models.load_model(path)
        
        #importa tamanho da entrada
        token = size.split(",")
        container["size"] = (int(token[0]), int(token[1]))

        #define os métodos de preprocessamento
        if type_model == self.TYPE_XCEPTION:
            container["preprocessing"] = keras.applications.xception.preprocess_input
        #as imaplementacoes usam a V2 da resnet
        elif type_model == self.TYPE_RESNET101:
            container["preprocessing"] = keras.applications.resnet_v2.preprocess_input
        elif type_model == self.TYPE_EFFICIENTNETB3:
            container["preprocessing"] = keras.applications.efficientnet.preprocess_input
        
        #caso o tipo nao esteja definido nenhum preprocessamento é aplicado
        else:
            container["preprocessing"] = self.fake_preprocess
        
        self.models[name] = container

    #retorna um modelo
    def configure_model(self, name, path, size, type_model):
        if not name in self.models:
            self._add_model(name, path, size, type_model)
        return self.models[name]
    
    def configure_classes(self, classes):
        self.CLASSES = classes
        self.CLASSES_NAMES = list(classes.keys())
    #pega modelo sem ativação para fazer o gradcam
    def get_model_without_activation(self, name, path):
        model = self.get_model(name, path)
        clone = tf.keras.models.clone_model(model)

        clone.layers[-1].activation = None
        return clone

    def fake_preprocess(self, img):
        return img

    def make_prediction(self, model, img_path):
        img_array = util.get_img_array(img_path, self.models[model]["size"], expand=True)
        img_array_preprocessed = self.models[model]["preprocessing"](img_array)
        prediction = self.models[model]["model"].predict(img_array_preprocessed)
        print(prediction.shape)
        class_index = np.argmax(prediction, axis=1)
        result={"class_name":self.CLASSES_NAMES[class_index[0]], "score":prediction[0, class_index[0]]}

        print(result)
        return result
