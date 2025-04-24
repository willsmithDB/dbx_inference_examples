# Distributed Model Inference Examples on Databricks with Huggingface and Accelerate

This repository provides tools and scripts for performing distributed model inference on Databricks using Huggingface and Accelerate. The focus is on leveraging data parallelism and model parallelism to efficiently utilize resources.

This is not official Databricks documentation nor official assets.

## Concepts

### Data Parallelism
Data parallelism involves splitting the input data across multiple devices, where each device processes a portion of the data independently. This approach is beneficial for scaling inference tasks as it allows simultaneous processing of data batches.

### Model Parallelism
Model parallelism involves splitting the model itself across multiple devices. This is useful for large models that cannot fit into the memory of a single device. By distributing the model, each device handles a portion of the model's layers or operations.

You can alsp have pipeline parallelism where we can end in as many inputs as we have GPUs whereas each part of will be working on their particular chunk. This results in more efficient usage of the GPUs rather than idling for certain model chunks during generation. 

![Model Parallelism](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/accelerate/pipeline_parallel.png)

For more details, refer to the [Huggingface Accelerate Distributed Inference Guide](https://huggingface.co/docs/accelerate/main/en/usage_guides/distributed_inference).
## Configuration

Before running the scripts, ensure you update the `config.yaml` file with the appropriate settings for your environment.

## Cluster Configurations

### General Cluster Config
- **DBR Version**: 16.3 ML: 16.3.x-gpu-ml-scala2.12
- **Instance Type**: Standard_NC48ads_A100_v4 [A100]
- **Memory**: 440GB
- **GPUs**: 2
- **VRAM**: 80GB x 2 GPUs

### Cluster Config for Llama 8b
- **DBR Version**: 16.3 ML: 16.3.x-gpu-ml-scala2.12
- **Instance Type**: Standard_NC12s_v3 [V100]
- **Memory**: 224GB
- **GPUs**: 2
- **VRAM**: 16GB x 2 GPUs

Ensure your cluster is configured according to the specifications above to achieve optimal performance.

## License

This project is licensed under the Apache License.