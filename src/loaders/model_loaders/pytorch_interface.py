"""Module containing an interface to trained PyTorch model_loaders."""

from loaders.model_loaders.base_model import BaseModel
import torch
import numpy as np

class PyTorchModel(BaseModel):

    def __init__(self, model=None, model_path='', backend='PYT'):
        """Init method

        :param model: trained PyTorch Model.
        :param model_path: path to trained model_loaders.
        :param backend: "PYT" for PyTorch framework.
        """

        super().__init__(model, model_path, backend)
        self.num_forward_pass = 0
        self.local_forward = 0

    def load_model(self):
        if self.model_path != '':
            self.model = torch.load(self.model_path)

    def get_output(self, input_tensor):
        """Expect 2d float array or tensor"""
        self.num_forward_pass += len(input_tensor)
        self.local_forward += len(input_tensor)
        return self.model(input_tensor).float()

    def get_pred_probs(self, input_tensor):
        """Support Function for LIME"""
        self.num_forward_pass += len(input_tensor)
        self.local_forward += len(input_tensor)
        return self.model(input_tensor).float()

    def predict_probs(self, input_instance, return_last=True, return_numpy=True):
        """Works 1d or 2d with numpy array or torch tensor
        Returns: np.array or torch.tensor of shape N(num_samples) x c (number of classes) or N x 1.
        """
        if torch.is_tensor(input_instance):
            input_instance = input_instance.float()
        else:
            input_instance = torch.tensor(input_instance.astype(np.float)).float()

        if len(input_instance.shape) == 1:
            ret = self.get_output(input_instance.unsqueeze(0))[0]
            if return_last:
                ret = ret[..., -1]
        elif len(input_instance.shape) == 2:
            ret = self.get_output(input_instance)
            if return_last:
                ret = ret[..., -1].unsqueeze(1)
        return ret.data.numpy() if return_numpy else ret

    def set_eval_mode(self):
        self.model.eval()

    def get_gradient(self, input):
        # Future Support
        raise NotImplementedError("Future Support")
