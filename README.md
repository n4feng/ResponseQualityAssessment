# Response Quality Assessment for Retrieval-Augmented Generation via Conditional Conformal Factuality

This repository contains production-ready code and resources for our research for paper "Response Quality Assessment for Retrieval-Augmented
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

-conf: configuration file location
-data
--out: Final subclaim with score output follo subclaims_schema
--processed: standard test data processed into base_schema
--raw: original raw data without structure difrectly from the source mentioned below
-index_store: store chunked documents and embeddings
-logs: store config used and log in format of run_{data}_{run_id} for each run
-src
--calibration: conformal prediction calibration logic
--common: some reusable component like config manager, faiss vector db manager
--data_processor: processor to convert raw QA data to standarlized data in this project (schema under data/processed)
--dataloader: load data from source data (like akariasai/popQA, kilt benchmark) to raw data
--rag: rag system support for document retrival
--subclaim_processor: handle subclaims for differnet dataset: gnerate subclaims from response, score, annotate subcliams.
--utils: other tools


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
Project is build on python version 3.11
First, set up project env using [requirements.txt](requirements.txt).
To run the pipeline:
```python
python main.py --config conf/config.yaml --dataset fact_score --query_size 500
```
Only 1 dataset at a time in 1 thread.
avaliable dataset currently are:
["fact_score", "hotpot_qa", "pop_qa", "medlf_qa"]

## More Information
For further details, please refer to our Paper: (To be uploaded to arXiv)
Please notice that the baseline group conditional conformal (https://arxiv.org/abs/2406.09714) result 
for medlfqav2 is produced through their codebase: github.com/jjcherian/conformal-safety
and is not in part of this repo.

## License

This project is licensed under the [MIT License](LICENSE).
