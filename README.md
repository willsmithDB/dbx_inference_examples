# Data and Model Parallelism with PyTorch for Large Language Models within Databricks 

This repository provides examples of data parallelism and model parallelism with PyTorch to demonstrate batch inferencing with large language models.
## Introduction

Large language models have achieved state-of-the-art results in various natural language processing (NLP) tasks. However, these models are often too large to fit in memory, and inferencing can be slow. To address this, we can use data parallelism and model parallelism to speed up batch inferencing.

## Data Parallelism
Data parallelism involves dividing the input data into smaller chunks and processing them in parallel across multiple devices (e.g., GPUs). This approach can significantly speed up batch inferencing.
### Example Code
The data_parallelism directory contains an example code that demonstrates data parallelism with PyTorch: