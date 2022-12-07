import logging
import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from time import perf_counter
import numpy as np
import torch
import csv
# Set up logging
logger = logging.getLogger(__name__)

def generate_sample_inputs(tokenizer, sequence_length):
  dummy_input = "dummy"
  embeddings = tokenizer(dummy_input, max_length=sequence_length, padding="max_length",return_tensors="pt")
  return tuple(embeddings.values())

def measure_latency(model, tokenizer, sequence_length):
    payload = generate_sample_inputs(tokenizer, sequence_length)
    latencies = []
    # warm up
    for _ in range(10):
        _ = model(*payload)
    # Timed run
    for _ in range(100):
        start_time = perf_counter()
        _ =  model(*payload)
        latency = perf_counter() - start_time
        latencies.append(latency)
    # Compute run statistics
    time_avg_ms = 1000 * np.mean(latencies)
    time_std_ms = 1000 * np.std(latencies)
    time_p95_ms = 1000 * np.percentile(latencies,95)
    return {"time_avg_ms": time_avg_ms, "time_std_ms": time_std_ms, "time_p95_ms": time_p95_ms, "sequence_length": sequence_length}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_neuron", action="store_true")
    parser.add_argument("--model_id", type=str)  
    parser.add_argument("--instance_type", type=str)  
    parser.add_argument("--sequence_length", type=int, default=None)
    
    # neuron specific args
    parser.add_argument("--num_neuron_cores", type=int, default=1)
    known_args, _ = parser.parse_known_args()  
    return known_args

def compile_model(model, tokenizer, sequence_length, num_neuron_cores):
    import torch.neuron
    payload = generate_sample_inputs(tokenizer, sequence_length)
    return torch.neuron.trace(model, payload)


def main(args):
  print(args)
  # load tokenizer and  model
  tokenizer = AutoTokenizer.from_pretrained(args.model_id)
  model = AutoModelForSequenceClassification.from_pretrained(args.model_id, torchscript=True)
  
  # compile model if neuron
  if args.is_neuron:
    model = compile_model(model, tokenizer, args.sequence_length, args.num_neuron_cores)
  
  # define sequence lengths to benchmark
  if args.sequence_length is None:
    sequence_lengths = [8,16,32,64,128, 256, 512] 
  else:
    sequence_lengths = [args.sequence_length]

  # benchmark model
  benchmark_dict = []
  for sequence_length in sequence_lengths:
    logger.info(f"Measuring latency for sequence length {sequence_length}")
    benchmark_dict.append({**measure_latency(model, tokenizer, sequence_length), 
      "instance_type": args.instance_type,
    })    
  
  # write results to csv
  keys = benchmark_dict[0].keys()
  with open(f'results/benchmmark_{args.instance_type}_{args.model_id.replace("-","_")}.csv', 'w', newline='') as output_file:
      dict_writer = csv.DictWriter(output_file, keys)
      dict_writer.writeheader()
      dict_writer.writerows(benchmark_dict)

if __name__ == "__main__":
  main(parse_args())
  
  
# python scripts/benchmark_transformers.py --model_id bert-base-uncased --instance_type c6i.2xlarge