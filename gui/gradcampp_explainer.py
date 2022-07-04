from numpy.core.fromnumeric import size
from base_explainer import BaseExplainer
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.cm as cm

class GradCAMPPExplainer(BaseExplainer):

    #implementacao do metodo abstrato
    def get_explanation(self, img, model, img_size, props, preprocess_input = None, index=None):
        #clona o modelo e remove a softmax da ultima camada
        clone = tf.keras.models.clone_model(model)
        clone.layers[-1].activation = None
        
        #cria modelo grad-cam
        grad_model = tf.keras.models.Model([clone.inputs], [clone.get_layer(props["conv_layer"]).output, clone.output])
        #transforma a imagem em array
        img_array = self.get_img_array(img, size = img_size, expand=False)
        #pre processa a imagem
        if preprocess_input:
            img_procecessed_array = preprocess_input(img_array)
        else:
            img_procecessed_array = img_array
        #faz o heatmap
        heatmap = self.__grad_cam_plus(grad_model, img_procecessed_array, props["conv_layer"], pred_index=index)
        #poe o heatmap na imagem
        heat, mask = self.__save_and_display_gradcam(img, heatmap)

        return heat


    #transforma a imagem em array
    def get_img_array(self, img_path, size, expand=True):
        # `img` is a PIL image of size 299x299
        img = keras.preprocessing.image.load_img(img_path, target_size=size)
        # `array` is a float32 Numpy array of shape (299, 299, 3)
        array = keras.preprocessing.image.img_to_array(img)
        # We add a dimension to transform our array into a "batch"
        # of size (1, 299, 299, 3)
        if expand:
            array = np.expand_dims(array, axis=0)
        return array

    #implementacao do grad-cam++
    def __grad_cam_plus(model, img, layer_name, category_id=None):
        img_tensor = np.expand_dims(img, axis=0)

        conv_layer = model.get_layer(layer_name)
        heatmap_model = tf.keras.models.Model([model.inputs], [conv_layer.output, model.output])

        with tf.GradientTape() as gtape1:
            with tf.GradientTape() as gtape2:
                with tf.GradientTape() as gtape3:
                    conv_output, predictions = heatmap_model(img_tensor)
                    if category_id==None:
                        category_id = np.argmax(predictions[0])
                    output = predictions[:, category_id]
                    conv_first_grad = gtape3.gradient(output, conv_output)
                conv_second_grad = gtape2.gradient(conv_first_grad, conv_output)
            conv_third_grad = gtape1.gradient(conv_second_grad, conv_output)

        global_sum = np.sum(conv_output, axis=(0, 1, 2))

        alpha_num = conv_second_grad[0]
        alpha_denom = conv_second_grad[0]*2.0 + conv_third_grad[0]*global_sum
        alpha_denom = np.where(alpha_denom != 0.0, alpha_denom, 1e-10)
        
        alphas = alpha_num/alpha_denom
        alpha_normalization_constant = np.sum(alphas, axis=(0,1))
        alphas /= alpha_normalization_constant

        weights = np.maximum(conv_first_grad[0], 0.0)

        deep_linearization_weights = np.sum(weights*alphas, axis=(0,1))
        grad_CAM_map = np.sum(deep_linearization_weights*conv_output[0], axis=2)

        heatmap = tf.maximum(grad_CAM_map, 0) / tf.math.reduce_max(grad_CAM_map)
        
        heatmap = heatmap.numpy()
        return heatmap

    def __save_and_display_gradcam(self, img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
        # Load the original image
        img = keras.preprocessing.image.load_img(img_path)
        img = keras.preprocessing.image.img_to_array(img)

        # Rescale heatmap to a range 0-255
        heatmap = np.uint8(255 * heatmap)
        im = Image.fromarray(heatmap)
        im = im.resize((img.shape[1], img.shape[0]))
        
        im = np.asarray(im)
        im = np.where(im > 0, 1, im)

        # Use jet colormap to colorize heatmap
        jet = cm.get_cmap("jet")

        # Use RGB values of the colormap
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]

    

        # Create an image with RGB colorized heatmap
        jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
        jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

        # Superimpose the heatmap on original image
        superimposed_img = jet_heatmap * alpha + img
        superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

        # Save the superimposed image
        #superimposed_img.save(cam_path)

        # Display Grad CAM
        #display(Image(cam_path))
        return superimposed_img, im