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
        # torch.repeat_interleave(tokens['token_type_ids'], batch_size, 0),
    )


# Build tokenizer and model
name = "textattack/distilbert-base-uncased-MRPC"
tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModelForSequenceClassification.from_pretrained(name, torchscript=True)

# Setup some example inputs
sequence_0 = "The company HuggingFace is based in New York City"
sequence_1 = "Apples are especially bad for your health"
sequence_2 = "HuggingFace's headquarters are situated in Manhattan"

paraphrase = encode(tokenizer, sequence_0, sequence_2)

not_paraphrase = encode(tokenizer, sequence_0, sequence_1)

# Run the original PyTorch model on examples
paraphrase_reference_logits = model(*paraphrase)[0]
not_paraphrase_reference_logits = model(*not_paraphrase)[0]

# Run precompiled TorchScript that is optimized by AWS Neuron Inf2
neuron_model = torch_neuronx.trace(model, paraphrase)

# Verify the TorchScript works on both example inputs
paraphrase_neuron_logits = neuron_model(*paraphrase)
not_paraphrase_neuron_logits = neuron_model(*not_paraphrase)

# compare results
print('Paraphrase Reference Logits:    ', paraphrase_reference_logits.detach().numpy())
print('Paraphrase Neuron Logits:       ', paraphrase_neuron_logits.detach().numpy())
print('Not-Paraphrase Reference Logits:', not_paraphrase_reference_logits.detach().numpy())
print('Not-Paraphrase Neuron Logits:   ', not_paraphrase_neuron_logits.detach().numpy())