# Results for Encoder Benchmark (BERT, RoBERTa) for CPU, Inferentia & Inferentia 2

Result tables

## `BERT` [bert-base-uncased](https://huggingface.co/bert-base-uncased) / p95 latency

| Sequence Length | CPU (c6i.2xlarge) | GPU A10G (5.2xlarge) | Inf1 (inf1.2xlarge) | Inf2 (inf2.2xlarge) | 
| --------------- | ----------------- | ------------------- | ------------------- | ------------------- |
| 8               | 19.9ms            | 8.2ms               | 6.9ms               | 1.1ms               |
| 16              | 23.1ms            | 8.2ms               | 8.2ms               | 1.2ms               |
| 32              | 27.7ms            | 8.1ms               | 7.1ms               | 1.3ms               |
| 64              | 36.2ms            | 8.2ms               |  7.2ms               | 1.6ms               |
| 128             | 54.3ms            | 8.1ms               | 8.1ms               | 2.4ms               |
| 256             | 108.6ms           | 8.1ms               | 9.8ms               | 4.4ms               |
| 512             | 226.1ms           | 9.9ms              | 18.4ms              | 10.7ms              |



## `BERT-large` [bert-large-uncased](https://huggingface.co/bert-large-uncased) / p95 latency

| Sequence Length | CPU (c6i.2xlarge) | GPU A10G (5.2xlarge) | Inf1 (inf1.2xlarge) | Inf2 (inf2.2xlarge) |
|-----------------|-------------------|----------------------|---------------------|---------------------|
| 8               | 65.54ms           | 15.81ms              |         23.91ms            | 3.5ms               |
| 16              | 117.7ms           | 15.97ms              |       23.80ms              | 3ms                 |
| 32              | 131.30ms          | 15.73ms              |       24.10ms              | 3.3ms               |
| 64              | 170.22ms          | 15.79ms              |      24.23ms               | 3.9ms               |
| 128             | 213.72ms          | 15.82ms              |           24.99ms          | 6.3ms               |
| 256             | 379.43ms          | 15.82ms              |          31.99ms           | 12.6ms              |
| 512             | 783.93ms          | 31.20ms              |          -           | 28.6ms              |


## `RoBERTa` [roberta-base](https://huggingface.co/roberta-base) / p95 latency 


| Sequence Length | CPU (c6i.2xlarge) | GPU A10G (5.2xlarge) | Inf1 (inf1.2xlarge) | Inf2 (inf2.2xlarge) |
|-----------------|-------------------|----------------------|---------------------|---------------------|
| 8               | 19.9ms            | 8.42ms               | 6.9ms               | 1.14ms              |
| 16              | 23.1ms            | 8.38ms               | 8.3ms               | 1.2ms               |
| 32              | 27.7ms            | 8.31ms               | 7.1ms               | 1.3ms               |
| 64              | 39.3ms            | 8.33ms               | 7.2ms               | 1.6ms               |
| 128             | 57.1ms            | 8.41ms               | 8.2ms               | 2.4ms               |
| 256             | 102.9ms           | 8.29ms               | 9.8ms               | 4.4ms               |
| 512             | 225.1ms           | 9.89ms               | 18.7ms              | 10.7ms              |


## DistilBERT [distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased) / p95 latency

| Sequence Length | CPU (c6i.2xlarge) | GPU A10G (5.2xlarge) | Inf1 (inf1.2xlarge) | Inf2 (inf2.2xlarge) |
|-----------------|-------------------|----------------------|---------------------|---------------------|
| 8               | 10ms              | 4.5ms                | 3.4ms               | 0.63ms              |
| 16              | 11ms              | 4.5ms                | 3.5ms               | 0.63ms              |
| 32              | 14.4ms            | 4.44ms               | 3.6ms               | 0.64ms              |
| 64              | 18.9ms            | 4.48ms               | 3.7ms               | 0.67ms              |
| 128             | 27.4ms            | 4.44ms               | 3.9ms               | 0.80ms              |
| 256             | 52.3ms            | 4.39ms               | 5.7ms               | 1.40ms              |
| 512             | 108.1ms           | 5.13ms               | 13.7ms              | 3.12ms              |

## AlBERT [albert-base-v2](https://huggingface.co/albert-base-v2) / p95 latency

| Sequence Length | CPU (c6i.2xlarge) | GPU A10G (5.2xlarge) | Inf1 (inf1.2xlarge) | Inf2 (inf2.2xlarge) |
|-----------------|-------------------|----------------------|---------------------|---------------------|
| 8               | 18.56ms           | 9.20ms               |  3.75ms                   | 1.37ms              |
| 16              | 26.66ms           | 9.25ms               |  3.78ms                   | 1.43ms              |
| 32              | 33.38ms           | 9.17ms               |  3.79ms                   | 1.42ms              |
| 64              | 46.11ms           | 9.19ms               |  4.0ms                   | 1.43ms              |
| 128             | 77.24ms           | 9.26ms               |  4.88ms                   | 1.72ms              |
| 256             | 148.67ms          | 9.19ms               |   8.33ms                  | 3.33ms              |
| 512             | 279.56ms          | 12.60ms              |   18.5ms                  | 7.55ms              |
# Vision models

## Vision Transformer [google/vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224) / p95 latency

| Image size | CPU (c6i.2xlarge) | GPU A10G (5.2xlarge) | Inf1 (inf1.2xlarge) | Inf2 (inf2.2xlarge) |
|------------|-------------------|----------------------|---------------------|---------------------|
| 3x224x224  | 91.3ms            | 7.3ms                |  9.48ms                   | 3.55ms              |



## Executions

**CPU**

```bash
python scripts/inference_transformers.py --model_id distilbert-base-uncased --instance_type c6i.2xlarge
```

**Inferentia**

```bash
python scripts/inference_transformers.py --model_id bert-large-uncased --instance_type inf2.xlarge --is_neuron 
```