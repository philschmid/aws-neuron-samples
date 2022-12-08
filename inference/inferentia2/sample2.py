import torch
import torch_neuronx
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import transformers


def encode(tokenizer, *inputs, max_length=128, batch_size=1):
    tokens = tokenizer.encode_plus(
        *inputs,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors="pt"
    )
    return (
        torch.repeat_interleave(tokens['input_ids'], batch_size, 0),
        torch.repeat_interleave(tokens['attention_mask'], batch_size, 0),
        # torch.repeat_interleave(tokens['token_type_ids'], batch_size, 0), # commented out since distilbert is not using them
    )


# Build tokenizer and model
name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModelForSequenceClassification.from_pretrained(name, torchscript=True)

# Setup some example inputs
sequence_0 = "I like you. I love you"

paraphrase = encode(tokenizer, sequence_0)

# Run the original PyTorch model on examples
paraphrase_reference_logits = model(*paraphrase)[0]

# Run precompiled TorchScript that is optimized by AWS Neuron Inf2
neuron_model = torch_neuronx.trace(model, paraphrase)

# Verify the TorchScript works on both example inputs
paraphrase_neuron_logits = neuron_model(*paraphrase)

# compare results
print('Reference Logits:    ', paraphrase_reference_logits.detach().numpy())
print('Neuron Logits:       ', paraphrase_neuron_logits.detach().numpy())