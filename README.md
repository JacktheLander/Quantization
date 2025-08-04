# Quantization

QAI Hub - Qualcomm's open-source library used to quantize models for a specific chipset

Linear Quantization - Simple and easy to use

Asymmetric Quantization - Good for activations that aren't centered at zero, because it uses the range dynamically

Symmetric Quantization - Centered at zero, but extremely fast due to the simple math and low memory usage

Per-Channel Quantization - Rather than quantizing the whole tensor we can quantize each output specifically for the highest accuracy with minimal performance drop

Per-Group Quantization - Best for maintaining precision, improves memory slightly, really for training at scale like LLMs

Inference Quantization - Using quantization for the weights in an activation

8-bit Quantizer - Performs an accelerated forward pass in the neural network

Quantize Layers - Replaces the linear layers in a model with quantized layers

Quantizing Models - Testing the Quantizer on an open-source LLM and Object Detection model, we see significant memory reduction with the same performance
