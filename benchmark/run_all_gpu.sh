# bert-base-uncased
python scripts/inference_transformers_gpu.py --model_id bert-base-uncased --instance_type g5.2xlarge

# bert-large-uncased
python scripts/inference_transformers_gpu.py --model_id bert-large-uncased --instance_type g5.2xlarge

# roberta-base
python scripts/inference_transformers_gpu.py --model_id roberta-base --instance_type g5.2xlarge

# distilbert-base-uncased
python scripts/inference_transformers_gpu.py --model_id distilbert-base-uncased --instance_type g5.2xlarge

# google/vit-base-patch16-224
python scripts/inference_transformers_vision_gpu.py --model_id google/vit-base-patch16-224 --instance_type g5.2xlarge

# albert-base-v2
python scripts/inference_transformers_gpu.py --model_id albert-base-v2 --instance_type g5.2xlarge
