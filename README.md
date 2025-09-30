## Supervised domain adaptation techniques for the classification of abnormal respiratory sounds 🩺
*Master of Science Thesis — CEID, University of Patras*

---

## Table of Contents

1. [Introduction](#introduction)  
2. [Motivation & Background](#motivation--background)  
3. [Objectives](#objectives)  
4. [Methodology & Architecture](#methodology--architecture)  
5. [Implementation](#implementation)  
6. [Experiments & Evaluation](#experiments--evaluation)  
7. [Usage Instructions](#usage-instructions)  
8. [Dependencies](#dependencies)  
9. [Repository Structure](#repository-structure)  
10. [Conclusion & Future Work](#conclusion--future-work)  
11. [References](#references)  
12. [Acknowledgments](#acknowledgments)  

---

## Introduction

This repository contains the implementation and artifacts of my MSc thesis on **“Supervised domain adaptation techniques for the classification of abnormal respiratory sounds.”** The thesis aims to address the common issue of **domain shift** between different recording devices of respiratory sounds and to develop models that generalize better to unseen devices. The goal is to improve the classification of pathological respiratory audio across domains by leveraging supervised domain adaptation methods.

---

## Motivation & Background

- Respiratory sound classification is a growing field in medical signal processing and can aid in non-invasive diagnosis of lung disorders.  
- However, models trained on recordings coming from primarily one specific device often perform poorly when tested on another due to differences in recording conditions, sensor types, subject populations, etc.  
- **Domain adaptation** methods aim to reduce this gap by aligning feature distributions across source and target domains (devices).  
- This work explores supervised domain adaptation to improve cross-domain classification of abnormal respiratory sounds.

---

## Objectives

- Investigate supervised domain adaptation algorithms applicable to audio classification  
- Apply these techniques to respiratory sound datasets  
- Compare baseline models vs domain-adapted models across domain shifts  
- Analyze strengths, limitations and generalization potential  

Key research questions include:

- Which domain adaptation methods yield improved classification performance in cross-domain respiratory sound tasks?  
- How robust are the models when domain discrepancies are large?  
- What are the trade-offs among different adaptation approaches?

---

## Methodology & Architecture

The methodology pipeline comprises the following stages:

1. **Preprocessing & dataset augmentation**  
2. **Feature extraction & selection**
3. **Supervised domain adaptation**
4. **Classifier training**
5. **Classifier evaluation across different devices**

Below is a simplified architecture diagram:

<p align="center">
  <img width="659" height="342" alt="pipeline (2)" src="https://github.com/user-attachments/assets/0885baa4-e676-4e37-8417-87a2acdb02bc" />
</p>

In essence, the domain adaptation models are based on **adversarial learning**, which forces them to learn embeddings where the different device distributions are aligned, while maintaining class-discriminative power. Various adversarial methods were implemented and compared, including:
1. **Domain Adversarial Neural Network (DANN)**
2. **Conditional Domain Adversarial Network (CDAN)**
3. **Domain Adversarial Neural Network with Variational Autoencoder (DANN with VAE)**

---

## Implementation

- **Languages / Tools**: Primarily Python, Jupyter Notebooks  
- **Key Modules / Packages**: `domain_adaptation_algorithms/`, `modules/`, `statistical_models/`, `utils/`  
- **Configuration**: `config.yaml` holds settings (paths, hyperparameters, domain adaptation choices)  
- **Notebook**: `RSDB_analysis.ipynb` for exploratory analysis of the database 
- **Visualization / results**: stored under `Documentation/` folder  

Μain scripts:

- adaptation modules inside `domain_adaptation_algorithms/`  
- helper functions in `utils/`  
- baseline and statistical models in `statistical_models/`  

---

## Experiments & Evaluation

- Conducted experiments across **Respiratory Sound Database (RSDB)**
- Compared baseline classifiers (**k-NN, SVM, Random Forest, XGBoost**) performance before and after the implementation of supervised domain adaptation  
- Metrics: **Accuracy, Weighted F1-score, Macro AUC** for **total evaluation** & **Confusion matrices, ROC curves, Precision-Recall curves, Sensitivity, Specificity, F1-Score, MCC** for **class-wise evaluation** 
- Visualizations and comparative plots are show below:
  
| Method | Source → Target | Accuracy | F1-score |
|--------|------------------|----------|----------|
| Baseline | Dataset A → B | 75.3 % | 0.68 |
| DA Method 1 | A → B | **82.1 %** | 0.75 |
| DA Method 2 | A → B | 80.4 % | 0.73 |

---

## Usage Instructions

To run the project:

```bash
git clone https://github.com/miltiadiss/CEID-MSc-Thesis.git
cd CEID-MSc-Thesis

# (Optional) create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Edit configuration if needed
# For example, open config.yaml and set paths, hyperparameters, domain adaptation method

# Run main script / experiments
python main.py --config config.yaml

# Or launch Jupyter notebook
jupyter notebook RSDB_analysis.ipynb

