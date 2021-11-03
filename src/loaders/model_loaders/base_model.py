"""Module containing a template class as an interface to ML model_loaders.
   Subclasses implement model_loaders interfaces for different ML frameworks such as TensorFlow or PyTorch.
   All model_loaders interface methods are in dice.model_loaders"""

class BaseModel:

    def __init__(self, model=None, model_path='', backend=''):
        """Init method

        :param model: trained ML Model.
        :param model_path: path to trained model_loaders.
        :param backend: ML framework. For frameworks other than TensorFlow or PyTorch, or for implementations other than standard DiCE (https://arxiv.org/pdf/1905.07697.pdf), provide both the module and class names as module_name.class_name. For instance, if there is a model_loaders interface class "SklearnModel" in module "sklearn_model.py" inside the subpackage dice.model_loaders, then backend parameter should be "sklearn_model.SklearnModel".
        """

        self.model = model
        self.model_path = model_path
        self.backend = backend

    def load_model(self):
        raise NotImplementedError

    def get_output(self):
        raise NotImplementedError

    def get_gradient(self):
        raise NotImplementedError
