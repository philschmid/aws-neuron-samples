
import os
import torch
import torch_neuronx
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# load tokenizer and model
# model_id = "distilbert-base-uncased-finetuned-sst-2-english"
model_id = "finiteautomata/bertweet-base-sentiment-analysis"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id, torchscript=True)

# create dummy input for max length 128
dummy_input = "I like you. I love you"
max_length = 128
embeddings = tokenizer(dummy_input, max_length=max_length, padding="max_length",return_tensors="pt")
neuron_inputs = tuple(embeddings.values())
# run through regular model
# with torch.inference_mode():
regular_output = model(**embeddings)[0]
print("vanilla output", regular_output)
regular_scores = torch.nn.Softmax(dim=-1)(regular_output)
regular_prediction = [{"label": model.config.id2label[item.argmax(-1).item()], "score": item.max().item()} for item in regular_scores]


# compile model with torch.neuron.trace and update config
model_neuron = torch_neuronx.trace(model, neuron_inputs)

# with torch.inference_mode():
output = model_neuron(*neuron_inputs)[0]
scores = torch.nn.Softmax(dim=-1)(output)
prediction = [{"label": model.config.id2label[item.argmax(-1).item()], "score": item.max().item()} for item in scores]

print("vanilla output", regular_output)
print("output",output)
print("vanilla scores", regular_scores)
print("scores",scores)
print("vanilla predictions:",regular_prediction)
print("predictions",prediction)


# vanilla predictions: [{'label': 'POSITIVE', 'score': 0.9998738765716553}]
# predictions [{'label': 'NEGATIVE', 'score': 0.5000945329666138}, {'label': 'NEGATIVE', 'score': 0.49990543723106384}]

# model.config.update({"traced_sequence_length": max_length})

# # save tokenizer, neuron model and config for later use
# save_dir="tmp"
# os.makedirs("tmp",exist_ok=True)
# model_neuron.save(os.path.join(save_dir,"neuron_model.pt"))
# tokenizer.save_pretrained(save_dir)
# model.config.save_pretrained(save_dir)
