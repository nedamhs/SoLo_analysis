# SoLo Analysis

This repository contains the code used for the technical validation and analysis of the SoLo dataset.

## Repository Structure

- `create_df.py` — constructs the combined dataframe across participants.
- `feature_extraction.py` — contains functions for extracting and aggregating AWARE, mobility, Samsung HRV, and Oura features.
- `technical_validation/` — contains notebooks and R scripts used for the technical validation analyses.

## Creating the Combined Dataframe

From the repository root, run:

```bash
python create_df.py --dataset-path /path/to/SoLo_dataset
```