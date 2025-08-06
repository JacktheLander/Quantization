### Testing the Quantizer on Open-Source LLMs ###
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
#from transformers import DetrImageProcessor, DetrForObjectDetection
from transformers import pipeline
from PIL import Image
import requests
import matplotlib as plt

from quantizer import W8A16LinearLayer
from quantize_layers import replace_linear_with_target_and_quantize
import sys
print(sys.version)
print(torch.__version__)

## Testing on TinyLLama LLM ##
model_id = "TinyLlama/TinyLlama_v1.1"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Generate story or response
pipe = pipeline("text-generation",
    model=model,
    tokenizer=tokenizer,
    do_sample=True,
    top_k=30,
    top_p=0.9,
    num_return_sequences=1,
    repetition_penalty=1.5)

prompt = "Do not go gentle into that good night, Old age should burn and rave at close of day; Rage, rage against the dying of the light."

print(prompt)
print(pipe(prompt, max_new_tokens=150, do_sample=True, temperature=0.7))
print("Model before:\n\n", model)

replace_linear_with_target_and_quantize(model, W8A16LinearLayer, ["lm_head"])
pipe.model
print(pipe(prompt, max_new_tokens=150, do_sample=True, temperature=0.7)[0]["generated_text"])
print("Model after:\n\n", model)
