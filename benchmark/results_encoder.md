# Results for Encoder Benchmark (BERT, RoBERTa) for CPU, Inferentia & Inferentia 2

Result tables

## `BERT` [bert-base-uncased](https://huggingface.co/bert-base-uncased) / p95 latency

| Sequence Length | CPU     | Inf1 | Inf2 |
| --------------- | ------- | ---- | ---- |
| 8               | 19.9ms  |      |      |
| 16              | 23.1ms  |      |      |
| 32              | 27.7ms  |      |      |
| 64              | 36.2ms  |      |      |
| 128             | 54.3ms  |      |      |
| 256             | 108.6ms |      |      |
| 512             | 226.1ms |      |      |


## `RoBERTa` (base) / p95 latency


| Sequence Length | CPU     | Inf1 | Inf2 |
| --------------- | ------- | ---- | ---- |
| 8               | 19.9ms  |      |      |
| 16              | 23.1ms  |      |      |
| 32              | 27.7ms  |      |      |
| 64              | 39.3ms  |      |      |
| 128             | 57.1ms  |      |      |
| 256             | 102.9ms |      |      |
| 512             | 225.1ms |      |      |


## DistilBERT (base-uncased) / p95 latency

| Sequence Length | CPU     | Inf1 | Inf2 |
| --------------- | ------- | ---- | ---- |
| 8               | 10ms    |      |      |
| 16              | 11ms    |      |      |
| 32              | 14.4ms  |      |      |
| 64              | 18.9ms  |      |      |
| 128             | 27.4ms  |      |      |
| 256             | 52.3ms  |      |      |
| 512             | 108.1ms |      |      |




## Executions

**CPU**

```bash
python scripts/inference_transformers.py --model_id distilbert-base-uncased --instance_type c6i.2xlarge
```

**Inferentia**

```bash
python scripts/inference_transformers.py --model_id bert-base-uncased --instance_type inf1.2xlarge --is_neuron 
```