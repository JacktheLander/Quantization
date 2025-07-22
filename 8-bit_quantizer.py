### Compressing a Model with 8-bit Precision ###

import torch
import torch.nn as nn
import torch.nn.functional as F

random_int8 = torch.randint(-128, 127, (32, 16)).to(torch.int8)
random_hs = torch.randn((1, 16), dtype=torch.bfloat16)
scales = torch.randn((1, 32), dtype=torch.bfloat16)
bias = torch.randn((1, 32), dtype=torch.bfloat16)

# The typical forward pass that we aim to quantize
print((F.linear(random_hs, random_int8.to(random_hs.dtype)) * scales) + bias)


## Cast the Weights to the Input Datatype ##
def w8_a16_forward(weight, input, scales, bias=None):
    
    casted_weights = weight.to(input.dtype)
    output = F.linear(input, casted_weights) * scales
    
    if bias is not None:
        output = output + bias
      
    return output

print("With bias:\n\n", w8_a16_forward(random_int8, random_hs, scales, bias))

print("\nWithout bias:\n\n", w8_a16_forward(random_int8, random_hs, scales))

## Stores Quantized Weights and Scales of the Model ##
class W8A16LinearLayer(nn.Module):
    # Initialize the Matrix  
    def __init__(self, in_features, out_features, bias=True, dtype=torch.float32):
        super().__init__()

        # The register buffer method lets us store our weights without computing gradients on the tensor
        self.register_buffer(
            "int8_weights",
            torch.randint(-128, 127, (out_features, in_features), dtype=torch.int8)
        )
        
        self.register_buffer("scales", torch.randn((out_features), dtype=dtype))
        
        if bias:
            self.register_buffer("bias", torch.randn((1, out_features), dtype=dtype))
        
        else:
            self.bias = None

    def forward(self, input):
        return w8_a16_forward(self.int8_weights, input, self.scales, self.bias)
      
    def quantize(self, weights):
        w_fp32 = weights.clone().to(torch.float32)

        scales = w_fp32.abs().max(dim=-1).values / 127
        scales = scales.to(weights.dtype)

        int8_weights = torch.round(weights
                        /scales.unsqueeze(1)).to(torch.int8)

        self.int8_weights = int8_weights
        self.scales = scales

module = W8A16LinearLayer(16, 32)
dummy_hidden_states = torch.randn(1, 6, 16)
print(module(dummy_hidden_states).shape, module(dummy_hidden_states).dtype)

module = W8A16LinearLayer(4, 8)
print("Weights before:\n" , module.int8_weights)

random_matrix = torch.randn((4, 8), dtype=torch.bfloat16)
module.quantize(random_matrix)
print("Weights After:\n" , module.int8_weights)

print(module.scales)
print(module.scales.shape)
print(module.int8_weights.shape)

# dequantized weights
module.int8_weights * module.scales.unsqueeze(1)

# original weights
random_matrix

# error
print((random_matrix - module.int8_weights * module.scales.unsqueeze(1)).abs().mean())
