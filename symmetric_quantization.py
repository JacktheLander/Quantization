### Using Uniform Scaling for Linear Quantization ###

import torch
from asymmetric_quantization import linear_q_with_scale_and_zero_point
from linear_quantization import plot_quantization_errors, linear_dequantization, 

def get_q_scale_symmetric(tensor, dtype=torch.int8):
    r_max = tensor.abs().max().item()
    q_max = torch.iinfo(dtype).max

    # return the scale
    return r_max/q_max

# test the implementation on a random 4x4 matrix
test_tensor = torch.randn((4, 4))
print(test_tensor)
print(get_q_scale_symmetric(test_tensor))

def linear_q_symmetric(tensor, dtype=torch.int8):
    scale = get_q_scale_symmetric(tensor)

    # in symmetric quantization zero point is = 0  
    quantized_tensor = linear_q_with_scale_and_zero_point(tensor, scale=scale, zero_point=0, dtype=dtype)
    
    return quantized_tensor, scale

quantized_tensor, scale = linear_q_symmetric(test_tensor)
dequantized_tensor = linear_dequantization(quantized_tensor,scale,0)
plot_quantization_errors(test_tensor, quantized_tensor, dequantized_tensor)
print(f"""Quantization Error : \{(dequantized_tensor - test_tensor).abs().square().mean()}""")
