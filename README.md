## Supervised domain adaptation techniques for the classification of abnormal respiratory sounds 🩺
*Master of Science Thesis — CEID, University of Patras*

<p align="center">
  <img width="491" height="192" alt="image" src="https://github.com/user-attachments/assets/fbcc3575-82c4-4e2d-9844-b0f986a6d7d9" />
</p>

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

---

## Introduction

This repository contains the implementation and artifacts of my MSc thesis on **“Supervised domain adaptation techniques for the classification of abnormal respiratory sounds.”** The thesis aims to address the common issue of **domain shift** between different recording devices of respiratory sounds and to develop models that generalize better to unseen devices. The goal is to improve the classification of pathological respiratory sounds (**crackles, wheezes**) across domains by leveraging supervised domain adaptation methods.

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

- Conducted experiments across **Respiratory Sound Database (RSDB)**: https://www.kaggle.com/datasets/vbookshelf/respiratory-sound-database 
- Compared baseline classifiers (**k-NN, SVM, Random Forest, XGBoost**) performance before and after the implementation of supervised domain adaptation  
- Metrics: **Accuracy, Weighted F1-score, Macro AUC** for **total evaluation** & **Confusion matrices, ROC curves, Precision-Recall curves, Sensitivity, Specificity, F1-Score, MCC** for **class-wise evaluation**

| Classifier | Accuracy | Macro AUC | Weighted F1-score |
|--------|----------|----------|----------|
| k-NN | 0.68 | 0.78 | 0.66 |
| **Non-Linear SVM** | 0.71 | 0.79 | 0.69 |
| Random Forest | 0.77 (+13.2%) | 0.86 | 0.77 |
| XGBoost | 0.74 | 0.85 | 0.74 |

**Best classifier**: Non-Linear SVM with **C=10.0, γ='scale'**

| Method | Accuracy | Macro AUC | Weighted F1-score |
|--------|----------|----------|----------|
| Baseline | 0.68 | 0.78 | 0.66 |
| DANN | 0.71 (+4.4%) | 0.79 | 0.69 |
| **CDAN** | **0.77 (+13.2%)** | **0.86** | **0.77** |
| DANN with VAE (joint training) | 0.74 (+8.8%) | 0.85 | 0.74 |
| DANN with VAE (sequential training) | 0.74 (+8.8%) | 0.84 | 0.73 |

**Best domain adaptation method**: CDAN with **λ=0.2**

<img width="367" height="329" alt="image" src="https://github.com/user-attachments/assets/355fc441-d491-44ce-9d50-81bd74e1df21" />
<img width="367" height="329" alt="image" src="https://github.com/user-attachments/assets/a793ac2a-6017-47d5-8160-a29d6f0c294e" />

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

---

## Repository structure

/
├── README.md  
├── config.yaml  
├── RSDB_analysis.ipynb  
├── .gitignore  
├── domain_adaptation_algorithms/  
│   ├── method1.py  
│   ├── method2.py  
│   └── …  
├── modules/  
│   ├── feature_extractor.py  
│   └── classifier.py  
├── statistical_models/  
│   └── baseline_models.py  
├── utils/  
│   └── helpers.py  
└── Documentation/  
    ├── pipeline.png  
    └── other figures, tables  


