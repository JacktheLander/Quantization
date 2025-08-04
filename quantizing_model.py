### Testing the Quantizer on Open-Source LLMs ###
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import requests

from 8-bit_quantizer import W8A16LinearLayer
from quantize_layers import replace_linear_with_target_and_quantize


## Testing on Salesforce's 350M Parameter LLM ##
model_id = "codegen-350M-mono"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)

tokenizer = AutoTokenizer.from_pretrained(model_id)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

print(pipe("def hello_world():", max_new_tokens=20, do_sample=False))
print("Model before:\n\n", model)

replace_linear_with_target_and_quantize(model, W8A16LinearLayer, ["lm_head"])
pipe.model
print(pipe("def hello_world():", max_new_tokens=20, do_sample=False)[0]["generated_text"])


## Testing on Facebook's Detr Resnet 50 Object Detection Model ##
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

previous_memory_footprint = model.get_memory_footprint()
print("Footprint of the model in MBs: ", previous_memory_footprint/1e+6)img_path = "dinner_with_friends.png"
image = Image.open(img_path).convert("RGB")
image.show()

inputs = processor(images=image, return_tensors="pt")
with torch.no_grad():
  outputs = model(**inputs)

# convert outputs (bounding boxes and class logits) to COCO API
# let's only keep detections with score > 0.9
target_sizes = torch.tensor([image.size[::-1]])
results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

def plot_results(model, pil_img, results):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    scores, labels, boxes = results["scores"], results["labels"], results["boxes"]
    colors = COLORS * 100
    for score, label, (xmin, ymin, xmax, ymax),c  in zip(scores.tolist(), labels.tolist(), boxes.tolist(), colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        text = f'{model.config.id2label[label]}: {score:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()

plot_results(model, image, results)
print(model)


replace_linear_with_target_and_quantize(model, W8A16LinearLayer, ["0", "1", "2", "class_labels_classifier"])
print(model)

inputs = processor(images=image, return_tensors="pt")
with torch.no_grad():
  outputs = model(**inputs)

# convert outputs (bounding boxes and class logits) to COCO API
# let's only keep detections with score > 0.9
target_sizes = torch.tensor([image.size[::-1]])
results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

plot_results(model, image, results)

new_footprint = model.get_memory_footprint()
print("Footprint of the model in MBs: ", new_footprint/1e+6)

# Memory saved
print("Memory saved in MBs: ", (previous_memory_footprint - new_footprint)/1e+6)
