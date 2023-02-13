# bert-base-uncased
python scripts/inference_transformers.py --model_id bert-base-uncased --instance_type c6i.2xlarge

# bert-large-uncased
python scripts/inference_transformers.py --model_id bert-large-uncased --instance_type c6i.2xlarge 

# roberta-base
python scripts/inference_transformers.py --model_id roberta-base --instance_type c6i.2xlarge 

# roberta-large
python scripts/inference_transformers.py --model_id roberta-large --instance_type c6i.2xlarge 

# distilbert-base-uncased
python scripts/inference_transformers.py --model_id distilbert-base-uncased --instance_type c6i.2xlarge 

# google/vit-base-patch16-224
python scripts/inference_transformers_vision.py --model_id google/vit-base-patch16-224 --instance_type c6i.2xlarge 

# albert-base-v2
python scripts/inference_transformers.py --model_id albert-base-v2 --instance_type c6i.2xlarge 
