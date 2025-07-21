### Grouping a Tensor with High Granularity to Maintain Accuracy ###

from per-channel_quantization import linear_q_symmetric_per_channel
from symmetric_quantization import get_q_scale_symmetric
from asymmetric_quantization import linear_q_with_scale_and_zero_point
from linear_quantization import plot_quantization_errors, linear_dequantization

def linear_q_symmetric_per_group(tensor, group_size, dtype=torch.int8):
    
    t_shape = tensor.shape
    assert t_shape[1] % group_size == 0
    assert tensor.dim() == 2
    
    tensor = tensor.view(-1, group_size)
    
    quantized_tensor, scale = linear_q_symmetric_per_channel(tensor, dim=0, dtype=dtype)
    quantized_tensor = quantized_tensor.view(t_shape)
    
    return quantized_tensor, scale

def linear_dequantization_per_group(quantized_tensor, scale, group_size):
    
    q_shape = quantized_tensor.shape
    quantized_tensor = quantized_tensor.view(-1, group_size)
    
    dequantized_tensor = linear_dequantization(quantized_tensor, scale, 0)
    dequantized_tensor = dequantized_tensor.view(q_shape)
    
    return dequantized_tensor

test_tensor = torch.rand((6, 6))
group_size = 3

quantized_tensor, scale = linear_q_symmetric_per_group(test_tensor, group_size=group_size)
dequantized_tensor = linear_dequantization_per_group(quantized_tensor, scale, group_size=group_size)
plot_quantization_errors(test_tensor, quantized_tensor, dequantized_tensor)
print(f"""Quantization Error : \{(dequantized_tensor - test_tensor).abs().square().mean())}""")
# Near perfect accuracy
