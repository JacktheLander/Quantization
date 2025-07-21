### Performing Quantization on Each Output Dimension Separately ###

import torch
from symmetric_quantization import get_q_scale_symmetric
from asymmetric_quantization import linear_q_with_scale_and_zero_point
from linear_quantization import plot_quantization_errors, linear_dequantization


def linear_q_symmetric_per_channel(r_tensor, dim, dtype=torch.int8):
    
    output_dim = r_tensor.shape[dim]
    # store the scales
    scale = torch.zeros(output_dim)

    # We iterate through rows and get the symmetric quantization scale for each
    for index in range(output_dim):
        sub_tensor = r_tensor.select(dim, index)
        scale[index] = get_q_scale_symmetric(sub_tensor, dtype=dtype)

    # reshape the scale
    scale_shape = [1] * r_tensor.dim()
    scale_shape[dim] = -1
    scale = scale.view(scale_shape)
    quantized_tensor = linear_q_with_scale_and_zero_point(r_tensor, scale=scale, zero_point=0, dtype=dtype)
   
    return quantized_tensor, scale

test_tensor=torch.tensor(
    [[191.6, -13.5, 728.6],
     [92.14, 295.5,  -184],
     [0,     684.6, 245.5]]
)

dim = 0 # Set to 0 to quantize along rows or 1 for columns
output_dim = test_tensor.shape[dim]
print(scale)


## Row Quantization ##

quantized_tensor_0, scale_0 = linear_q_symmetric_per_channel(test_tensor, dim=0)
dequantized_tensor_0 = linear_dequantization(quantized_tensor_0, scale_0, 0)

plot_quantization_errors(test_tensor, quantized_tensor_0, dequantized_tensor_0)
print(f"""Quantization Error : \{(dequantized_tensor_0 - test_tensor).abs().square().mean())}""")
# Outperforms per tensor symmetric quantization


## Column Quantization ##

quantized_tensor_1, scale_1 = linear_q_symmetric_per_channel(test_tensor, dim=1)
dequantized_tensor_1 = linear_dequantization(quantized_tensor_1, scale_1, 0)

plot_quantization_errors(test_tensor, quantized_tensor_1, dequantized_tensor_1, n_bits=8)
print(f"""Quantization Error : \{(dequantized_tensor_1 - test_tensor).abs().square().mean())}""")
# Is even more accurate than symmetric quantization by row
