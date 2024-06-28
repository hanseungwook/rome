import torch

class ModelWithRef(torch.nn.Module):
    def __init__(self, model, ref_model):
        super().__init__()
        self.model = model
        self.ref_model = ref_model

    def forward(self, ref=False, *args, **kwargs):
        if ref:
            return self.ref_model(*args, **kwargs)
        else:
            return self.model(*args, **kwargs)
    
    def generate(self, ref=False, *args, **kwargs):
        if ref:
            return self.ref_model.generate(*args, **kwargs)
        else:
            return self.model.generate(*args, **kwargs)