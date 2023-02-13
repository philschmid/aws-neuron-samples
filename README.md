# Training and Inference Examples for Hugging Face Transformers and AWS Neuron (`Inf1`, `Inf2`, `Trn1`)

This repository contains instructions/examples/tutorials using the `neuron-sdk` for running inference and training Hugging Face libraries like [transformers](https://huggingface.co/docs/transformers/index), [datasets](https://huggingface.co/docs/datasets/index) with AWS Accelerators including [AWS Inferentia](https://aws.amazon.com/de/machine-learning/inferentia/) & [AWS Trainium](https://aws.amazon.com/de/machine-learning/trainium/).

### Inference

* [Speed up BERT inference with Hugging Face Transformers and AWS Inferentia](https://www.philschmid.de/huggingface-bert-aws-inferentia)
* [Accelerate Hugging Face Transformers with AWS Inferentia 2 | Benchmarking Transformers for AWS Inferentia 2]()


### Training

* 

## Resources

* [Github: aws-neuron-sdk](https://github.com/aws-neuron/aws-neuron-sdk)
* [Neuron Documentation](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/index.html)
* [Getting Started Neuron](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/quick-start/torch-neuron.html)

## Requirements

Before we can start make sure you have met the following requirements

* AWS Account with quota for `Inf1`, `Inf2`, `Trn1` instances.
* [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) installed
* AWS IAM user [configured in CLI](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html) with permission to create and manage ec2 instances


## Start environment

inf2:

```bash
source ../aws_neuron_venv_pytorch_p37/bin/activate
```
