from abc import ABC, abstractmethod

class BaseExplainer(ABC):

    @abstractmethod
    def get_explanation(self, img, model, img_size, props, preprocess_input = None, index=None):
        pass