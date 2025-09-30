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

This repository contains the implementation and artifacts of my MSc thesis on **“Supervised domain adaptation techniques for the classification of abnormal respiratory sounds.”** The thesis aims to address the common issue of **domain shift** between different recording devices of respiratory sounds and to develop models that generalize better to unseen devices. The goal is to improve the classification of pathological respiratory audio across domains by leveraging domain adaptation methods.

---

## Motivation & Background

- Respiratory sound classification is a growing field in medical signal processing and can aid in non-invasive diagnosis of lung disorders.  
- However, models trained on one dataset often perform poorly when tested on another due to differences in recording conditions, sensor types, subject populations, etc.  
- **Domain adaptation** methods aim to reduce this gap by aligning feature distributions across source and target domains.  
- This work explores supervised domain adaptation (i.e. when some labeled data in target domain is available) to improve cross-domain classification of abnormal respiratory sounds.

---

## Objectives

- Investigate supervised domain adaptation algorithms applicable to audio classification  
- Apply these techniques to respiratory sound datasets  
- Compare baseline models vs domain-adapted models across domain shifts  
- Analyze strengths, limitations, and generalization potential  

Key research questions include:

- Which domain adaptation methods yield improved classification performance in cross-domain respiratory sound tasks?  
- How robust are the models when domain discrepancies are large?  
- What are the trade-offs (e.g., complexity, convergence) among different adaptation approaches?

---

## Methodology & Architecture

The methodology pipeline comprises the following stages:

1. **Preprocessing & feature extraction**  
2. **Baseline classifier training**  
3. **Domain adaptation module**  
4. **Evaluation across domains**

Below is a simplified architecture diagram:

<p align="center">
  <img width="859" height="442" alt="pipeline (2)" src="https://github.com/user-attachments/assets/0885baa4-e676-4e37-8417-87a2acdb02bc" />
</p>

In essence, the model learns embeddings where source and target distributions are aligned, while maintaining class-discriminative power.

Various supervised domain adaptation methods (e.g. discrepancy-based, adversarial) were implemented and compared.

---

## Implementation

- **Languages / Tools**: Primarily Python, Jupyter Notebooks  
- **Key Modules / Packages**: `domain_adaptation_algorithms/`, `modules/`, `statistical_models/`, `utils/`  
- **Configuration**: `config.yaml` holds settings (paths, hyperparameters, domain adaptation choices)  
- **Notebook**: `RSDB_analysis.ipynb` for exploratory and result analysis  
- **Visualization / results**: stored under `Documentation/` and output folders  

Examples of main scripts:

- `main.py` (if applicable)  
- adaptation modules inside `domain_adaptation_algorithms/`  
- helper functions in `utils/`  
- baseline and statistical models in `statistical_models/`  

---

## Experiments & Evaluation

- Conducted experiments across multiple respiratory sound datasets (different domains)  
- Compared baseline classifiers (without adaptation) vs models with supervised domain adaptation  
- Metrics: accuracy, F1-score, confusion matrices, domain discrepancy measures  
- Visualizations and comparative plots are included in the documentation  

> **Note**: You should provide numeric tables and plots here once finalized—for example:

| Method | Source → Target | Accuracy | F1-score |
|--------|------------------|----------|----------|
| Baseline | Dataset A → B | 75.3 % | 0.68 |
| DA Method 1 | A → B | **82.1 %** | 0.75 |
| DA Method 2 | A → B | 80.4 % | 0.73 |

Include confusion matrices, ROC curves, etc.

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

