
import os
import tensorflow  # to workaround a protobuf version conflict issue
import torch
import torch.neuron
from transformers import AutoTokenizer, AutoModelForSequenceClassification

num_cores = 4 # This value should be 4 on inf1.xlarge and inf1.2xlarge
os.environ['NEURON_RT_NUM_CORES'] = str(num_cores)

# load tokenizer and model
model_id = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id, torchscript=True)

# create dummy input for max length 128
dummy_input = "I like you. I love you"
max_length = 128
embeddings = tokenizer(dummy_input, max_length=max_length, padding="max_length",return_tensors="pt")
neuron_inputs = tuple(embeddings.values())

# compile model with torch.neuron.trace and update config
model_neuron = torch.neuron.trace(model, neuron_inputs)

with torch.inference_mode():
  res = model_neuron(*neuron_inputs)[0]
  print(res.softmax(-1))


# model.config.update({"traced_sequence_length": max_length})

# # save tokenizer, neuron model and config for later use
# save_dir="tmp"
# os.makedirs("tmp",exist_ok=True)
# model_neuron.save(os.path.join(save_dir,"neuron_model.pt"))
# tokenizer.save_pretrained(save_dir)
# model.config.save_pretrained(save_dir)
