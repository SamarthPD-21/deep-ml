import torch

def to_float_tensor(values):
    # TODO: return a torch.float32 tensor built from `values`
    X= torch.tensor(values)
    return X.to(torch.float32)
    pass
