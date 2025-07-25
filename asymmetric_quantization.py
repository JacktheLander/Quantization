### Finding the Scale and Zero Point for Linear Quantization ###

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import torch
from linear_quantization import linear_q_with_scale_and_zero_point, linear_dequantization, plot_quantization_errors

# a dummy tensor to test the implementation
test_tensor=torch.tensor(
    [[191.6, -13.5, 728.6],
     [92.14, 295.5,  -184],
     [0,     684.6, 245.5]]
)

q_min = torch.iinfo(torch.int8).min
q_max = torch.iinfo(torch.int8).max]
print(q_min)
print(q_max)

# r_min = test_tensor.min()
r_min = test_tensor.min().item()
print(r_min)

r_max = test_tensor.max().item()
print(r_max)

scale = (r_max - r_min) / (q_max - q_min)
print(scale)

zero_point = q_min - (r_min / scale)
print(zero_point)

zero_point = int(round(zero_point))
print(zero_point)


def get_q_scale_and_zero_point(tensor, dtype=torch.int8):
    
    q_min, q_max = torch.iinfo(dtype).min, torch.iinfo(dtype).max
    r_min, r_max = tensor.min().item(), tensor.max().item()

    scale = (r_max - r_min) / (q_max - q_min)

    zero_point = q_min - (r_min / scale)

    # clip the zero_point to fall in [quantized_min, quantized_max]
    if zero_point < q_min:
        zero_point = q_min
    elif zero_point > q_max:
        zero_point = q_max
    else:
        # round and cast to int
        zero_point = int(round(zero_point))
    
    return scale, zero_point

new_scale, new_zero_point = get_q_scale_and_zero_point(test_tensor)
print(new_scale)
print(new_zero_point)


## Using them for Quantization and Dequantization ##

quantized_tensor = linear_q_with_scale_and_zero_point(test_tensor, new_scale, new_zero_point)
dequantized_tensor = linear_dequantization(quantized_tensor, new_scale, new_zero_point)
plot_quantization_errors(test_tensor, quantized_tensor, dequantized_tensor)

print("MSE:", (dequantized_tensor-test_tensor).square().mean())


## Complete Linear Quantizer ##

def linear_quantization(tensor, dtype=torch.int8):
    scale, zero_point = get_q_scale_and_zero_point(tensor, dtype=dtype)
    
    quantized_tensor = linear_q_with_scale_and_zero_point(tensor, scale, zero_point, dtype=dtype)
    
    return quantized_tensor, scale , zero_point

# Test with a random tensor
r_tensor = torch.randn((4, 4))
print(r_tensor)

quantized_tensor, scale, zero_point = linear_quantization(r_tensor)
print(quantized_tensor, scale, zero_point)

dequantized_tensor = linear_dequantization(quantized_tensor, scale, zero_point)
plot_quantization_errors(r_tensor, quantized_tensor, dequantized_tensor)
print("MSE:",(dequantized_tensor-r_tensor).square().mean())
