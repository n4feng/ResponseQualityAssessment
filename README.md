# Response Quality Assessment for Retrieval-Augmented Generation via Conditional Conformal Factuality

This repository contains code and resources for our research for paper "Response Quality Assessment for Retrieval-Augmented
Generation via Conditional Conformal Factuality" Accepted by SIGIR 2025

Project is build on python version 3.11

## Table of Contents
- [Structure](#Structure)
- [Data](#data)
  - [Query Data](#query-data)
  - [Wikipedia Extraction](#wikipedia-extraction)
- [Usage](#usage)
- [References](#references)
- [More Information](#more-information)

## Structure

-conf: configuration file location
-data
--out: Final subclaim with score output follo subclaims_schema
--processed: standard test data processed into base_schema
--raw: original raw data without structure difrectly from the source mentioned below
--index_store: store chunked documents 

## Data
### Query Data
This repository includes the following query datasets:
- [FactScore](https://github.com/shmsw25/FActScore)
- [PopQA](https://huggingface.co/datasets/akariasai/PopQA)
- [HotpotQA](https://huggingface.co/datasets/hotpotqa/hotpot_qa)
- [MedLFQA] (https://github.com/dmis-lab/OLAPH/tree/main/MedLFQA) or (https://github.com/jjcherian/conformal-safety/tree/main/data/MedLFQAv2)

### Wikipedia Extraction
We utilize Wikipedia dumps for knowledge retrieval:
- [enwiki-20230401.db](https://drive.google.com/file/d/1mekls6OGOKLmt7gYtHs0WGf5oTamTNat/view?usp=drive_link)
This file is not included in this github, you could download it through ming's google drive above (source: https://github.com/shmsw25/FActScore) and put it under
\data\raw folder in order to generate reference doucument for wiki based queries (popqa and hotpotqa)

## Usage
First, set up project env using [requirements.txt](requirements.txt).
To run the pipeline:
```python
python main.py --config conf/config.yaml --dataset dragonball --query_size 500
```


## More Information
For further details, please refer to our Paper: (To be uploaded to arXiv)

