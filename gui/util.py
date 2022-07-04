import tensorflow.keras as keras
import numpy as np

#transforma a imagem em array
def get_img_array(img_path, size, expand=True):
    # `img` is a PIL image of size 299x299
    img = keras.preprocessing.image.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    if expand:
        array = np.expand_dims(array, axis=0)
    return array