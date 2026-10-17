import torch

def add_bias(x: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # TODO: add b to every row of x using broadcasting
    for i in range(len(x)):
        for j in range(len(x[i])):
            x[i][j] = x[i][j]+b[j]

    return x
    pass
