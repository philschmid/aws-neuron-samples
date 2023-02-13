# bert-base-uncased
python scripts/inference_transformers.py --model_id bert-base-uncased --instance_type inf2.xlarge --is_neuron 

# bert-large-uncased
python scripts/inference_transformers.py --model_id bert-large-uncased --instance_type inf2.xlarge --is_neuron

# roberta-base
python scripts/inference_transformers.py --model_id roberta-base --instance_type inf2.xlarge --is_neuron

# distilbert-base-uncased
python scripts/inference_transformers.py --model_id distilbert-base-uncased --instance_type inf2.xlarge --is_neuron

# google/vit-base-patch16-224
python scripts/inference_transformers_vision.py --model_id google/vit-base-patch16-224 --instance_type inf2.xlarge --is_neuron

# albert-base-v2
python scripts/inference_transformers.py --model_id albert-base-v2 --instance_type inf2.xlarge --is_neuron