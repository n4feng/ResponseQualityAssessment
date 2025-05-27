# Response Quality Assessment for Retrieval-Augmented Generation via Conditional Conformal Factuality

This repository contains code and resources for our research for paper "Response Quality Assessment for Retrieval-Augmented
Generation via Conditional Conformal Factuality" Accepted by SIGIR 2025

## Table of Contents
- [Structure](#Structure)
- [Data](#data)
  - [Query Data](#query-data)
  - [Wikipedia Extraction](#wikipedia-extraction)
- [Usage](#usage)
- [References](#references)
- [More Information](#more-information)

## Structure


## Data
### Query Data
This repository includes the following query datasets:
- [FactScore](https://arxiv.org/abs/2305.14251)
- [PopQA](https://huggingface.co/datasets/akariasai/PopQA)
- [HotpotQA](https://huggingface.co/datasets/hotpotqa/hotpot_qa)
- [MedLFQA](https://github.com/jjcherian/conformal-safety/tree/main/data/MedLFQAv2)

### Wikipedia Extraction
We utilize Wikipedia dumps for knowledge retrieval:
- [enwiki-20230401.db](https://drive.google.com/file/d/1mekls6OGOKLmt7gYtHs0WGf5oTamTNat/view?usp=drive_link)

## Usage
First, set up project env using [requirements.txt](requirements.txt).
To run the pipeline:
```python
python main.py --config conf/config.yaml --dataset dragonball --query_size 500
```


## More Information
For further details, please refer to our Paper: (To be uploaded to arXiv)

