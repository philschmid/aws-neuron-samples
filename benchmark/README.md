# Benchmark AWS Accelerators

This directory contains a set of benchmarks for the AWS Accelerators. The benchmarks are written in Python. The benchmarks are designed to be run on a single machine, and are not designed to be run in parallel. 

**Available Benchmarks**

* [inference_transformers.py](scripts/inference_transformers.py) to benchmark encoder inference for CPU, Inf1 and Inf2. The reults are saved as a CSV file. Results and previous runs can be found in [Result inference Inferentia 2](results_encoder.md)


## How to run the benchmarks

install packages with `pip install -r [cpu|inf]_requirements.txt`

Run the benchmark

```bash
python scripts/inference_transformers.py --model_name_or_path bert-base-uncased --instance_type inf1.xlarge --is_neuron 
```