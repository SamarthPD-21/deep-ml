import torch
import torch.nn as nn

def single_neuron_forward(x: torch.Tensor) -> float:
    # 1. Create the linear layer with 3 inputs and 1 output
    neuron = nn.Linear(3, 1)
    
    # 2. Fix the parameters under torch.no_grad() using in-place copies (.copy_())
    with torch.no_grad():
        neuron.weight.copy_(torch.tensor([[0.5, -0.2, 0.3]]))
        neuron.bias.copy_(torch.tensor([0.1]))
        
    # 3. Run the forward pass on input x
    out_tensor = neuron(x)
    
    # 4. Extract the scalar tensor value and convert it to a Python float
    return out_tensor.item()
