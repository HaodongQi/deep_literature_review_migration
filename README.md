# Exploring Migration Research with Large Language Models 

## Overview

The ability of migration scholars to synthesize knowledge is increasingly hindered by the rapid expansion of the field — both in volume and interdisciplinarity.

To address this challenge, we develop LlaMig (Large Language Model for Migration Research), an open-source, locally deployable framework designed to transform large volumes of scholarly text into a structured, queryable database.
This repository contains:

- Scripts for fine-tuning Llama-3.2-3B
- Exploratory analysis of LlaMig’s classifications of published migration articles
- Tools for mapping trends and themes in migration research
 
## Objectives

This repository provides tools and scripts to support:
### 📊 Analysis of literature datasets
- Mapping trends and thematic patterns in migration research
- Identifying critical research gaps

### 🤖 Fine-tuning open-source large language models
- Supporting automated literature review tasks
- Improving classification accuracy for migration scholarship

## Repository Structure
The project is structured to clearly separate:
- **analysis/** – Exploratory analysis and classification evaluation 
- **fine-tuning/** – Model training pipelines and configurations

### 📂 analysis/
Contains exploratory scripts, visualizations, and LlaMig classification outputs used to understand and evaluate the literature dataset.

#### Key scripts:
- analysis_FT_HQ_github_V1.R
Evaluates the performance of the fine-tuned model on migration literature classification tasks.
- analysis_trends_HQ_github_v1.R
Generates temporal and thematic insights into migration research trends.
- preliminary-classification-L32-3B.py
Performs automated initial classification using the original Llama-3.2-3B model.

#### This directory also contains:
- Classification datasets (train/test splits)
- Accuracy evaluation outputs
- Visualization files (e.g., trend plots, heatmaps, network graphs)

### 📂 fine-tuning/
Contains model training code and configuration files for fine-tuning open-source large language models.

This includes:
- Training pipelines
- Configuration files
- Dataset preparation scripts

## Conceptual Workflow
1.	📚 Collect and clean migration literature datasets
2.	🤖 Perform preliminary LLM classification
3.	🔁 Fine-tune Llama-3.2-3B on domain-specific data
4.	📊 Evaluate classification accuracy
5.	📈 Analyze trends and thematic developments
 
## Future Development
- Expanding labeled training data
- Improving classification robustness
- Adding queryable database features


