import logging
import argparse
from transformers import AutoImageProcessor, AutoModelForImageClassification
from time import perf_counter
import numpy as np
import torch
import csv
import os
from datasets import load_dataset

# Set up logging
logger = logging.getLogger(__name__)

def generate_sample_inputs(processor):
  dataset = load_dataset("huggingface/cats-image")
  image = dataset["test"]["image"][0]
  embeddings = processor(image,return_tensors="pt")
  print(embeddings["pixel_values"].shape)
  return tuple(embeddings.values())
  

def measure_latency(model, input):
    latencies = []
    # warm up
    for _ in range(10):
        _ = model(*input)
    # Timed run
    for _ in range(100):
        start_time = perf_counter()
        _ =  model(*input)
        latency = perf_counter() - start_time
        latencies.append(latency)
    # Compute run statistics
    time_avg_ms = 1000 * np.mean(latencies)
    time_std_ms = 1000 * np.std(latencies)
    time_p95_ms = 1000 * np.percentile(latencies,95)
    return {"time_avg_ms": time_avg_ms, "time_std_ms": time_std_ms, "time_p95_ms": time_p95_ms}


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

def compile_model_inf1(model, payload, num_neuron_cores):
    os.environ['NEURON_RT_NUM_CORES'] = str(num_neuron_cores)
    import torch.neuron
    return torch.neuron.trace(model, payload)

def compile_model_inf2(model, payload, num_neuron_cores):
    # use only one neuron core
    os.environ["NEURON_RT_NUM_CORES"] = str(num_neuron_cores)
    import torch_neuronx
    return torch_neuronx.trace(model, payload)


def main(args):
  print(args)

  # benchmark model
  benchmark_dict = []
  # load processor and  model
  processor = AutoImageProcessor.from_pretrained(args.model_id)
  model = AutoModelForImageClassification.from_pretrained(args.model_id, torchscript=True)
  
  # generate sample inputs
  payload = generate_sample_inputs(processor)

  # compile model if neuron
  if args.is_neuron:
    if "inf1" in args.instance_type:
      model = compile_model_inf1(model,payload, args.num_neuron_cores)
    elif "inf2" in args.instance_type:
      model = compile_model_inf2(model, payload, args.num_neuron_cores)
    else:
      raise ValueError("Unknown neuron version")

    
  
  res = measure_latency(model, payload)
  print(res)
  benchmark_dict.append({**res,"instance_type": args.instance_type})    
  
  # write results to csv
  keys = benchmark_dict[0].keys()
  with open(f'results/benchmmark_{args.instance_type}_{args.model_id.split("/")[-1].replace("-","_")}.csv', 'w', newline='') as output_file:
      dict_writer = csv.DictWriter(output_file, keys)
      dict_writer.writeheader()
      dict_writer.writerows(benchmark_dict)

if __name__ == "__main__":
  main(parse_args())
  
  
# python scripts/benchmark_transformers.py --model_id bert-base-uncased --instance_type c6i.2xlarge
# {'time_avg_ms': 8.10589524991883, 'time_std_ms': 0.09509256634579266, 'time_p95_ms': 8.25341524941905, 'sequence_length': 128}
# {'time_avg_ms': 7.0798250301595544, 'time_std_ms': 0.07013446319476516, 'time_p95_ms': 7.2283735508790405, 'sequence_length': 64}
# {'time_avg_ms': 7.0568497200838465, 'time_std_ms': 0.06201203367767892, 'time_p95_ms': 7.158065150815674, 'sequence_length': 32}
# {'time_avg_ms': 8.227177910039245, 'time_std_ms': 0.05096229434436981, 'time_p95_ms': 8.318828549545287, 'sequence_length': 16}
# {'time_avg_ms': 6.88982284003032, 'time_std_ms': 0.03838955933742761, 'time_p95_ms': 6.972277099521307, 'sequence_length': 8}