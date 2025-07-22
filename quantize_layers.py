### Using the 8-bit Quantizer to Replace the Linear Layers in a Model ###

import torch
import torch.nn as nn

from 8-bit_quantizer import W8A16LinearLayer

## We replace linear layers with the quantized layer for faster inference ##
def replace_linear_with_target(module, target_class, module_name_to_exclude):
    # Find linear layers in the model
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and not any([x == name for x in module_name_to_exclude]):
            old_bias = child.bias

            new_module = target_class(child.in_features, child.out_features, old_bias is not None, child.weight.dtype)
            
            setattr(module, name, new_module)
            if old_bias is not None:
              getattr(module, name).bias = old_bias
        else:
            # Recursively call the function for nested modules
            replace_linear_with_target(child, target_class, module_name_to_exclude)


# A simple model to test the replacement
class DummyModel(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.emb = torch.nn.Embedding(1, 1)
    # Try with bias
    self.linear_1 = nn.Linear(1, 1)
    # Try without bias
    self.linear_2 = nn.Linear(1, 1, bias=False)
    # Lm prediction head
    self.lm_head = nn.Linear(1, 1, bias=False)

model_1 = DummyModel()
model_2 = DummyModel()

# Test replacing all linear layers vs excluding the output layer
replace_linear_with_target(model_1, W8A16LinearLayer, ["lm_head"])
print(model_1)

replace_linear_with_target(model_2, W8A16LinearLayer, [])
print(model_2)


## Here we quantize the weights as well reducing model size ##
def replace_linear_with_target_and_quantize(module, target_class, module_name_to_exclude):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and not any([x == name for x in module_name_to_exclude]):
            old_bias = child.bias
            old_weight = child.weight

            new_module = target_class(child.in_features, child.out_features, old_bias is not None, child.weight.dtype)
            setattr(module, name, new_module)
            getattr(module, name).quantize(old_weight)
            
            if old_bias is not None:
              getattr(module, name).bias = old_bias
        else:
            # Recursively call the function for nested modules
            replace_linear_with_target_and_quantize(child, target_class, module_name_to_exclude)

model_3 = DummyModel()
replace_linear_with_target_and_quantize(model_3, W8A16LinearLayer, ["lm_head"])
print(model_3)
