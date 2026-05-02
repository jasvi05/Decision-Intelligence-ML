<div align="center">

# Decision Intelligence ML

### A collection of 3 real-world machine learning projects that automate high-stakes business decisions — from hiring and sales forecasting to customer support automation.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![NLP](https://img.shields.io/badge/NLP-Text%20Classification-4CAF50?style=for-the-badge)](#)
[![Time Series](https://img.shields.io/badge/Time%20Series-Forecasting-9C27B0?style=for-the-badge)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

</div>

---

## Overview

**Decision Intelligence ML** is a monorepo of three end-to-end machine learning projects, each targeting a critical business decision-making problem across different domains — **HR, Sales, and Customer Support**.

Each project is self-contained with its own dataset, preprocessing pipeline, model training, and evaluation — demonstrating how machine learning can replace slow, biased, or manual decision workflows with fast, consistent, and data-driven automation.

---

## Repository Structure

```
Decision-Intelligence-ML/
│
├── resume-screening-system/        # ML-powered resume classifier for HR automation
│
├── sales-forecasting/              # Time series model for future sales prediction
│
└── support-ticket-classification/  # NLP classifier for customer support routing
```

---

## Projects at a Glance

| # | Project | Domain | ML Type | Key Technique |
|---|---|---|---|---|
| 1 | [Resume Screening System](#-1-resume-screening-system) | HR / Recruitment | Supervised | NLP + Text Classification |
| 2 | [Sales Forecasting](#-2-sales-forecasting) | Business / Retail | Supervised | Time Series Regression |
| 3 | [Support Ticket Classification](#-3-support-ticket-classification) | Customer Support | Supervised | NLP + Multi-class Classification |

---

## 1. Resume Screening System

### *Automating the first step of hiring with NLP and Machine Learning*

#### Problem
HR teams manually screen hundreds of resumes per job opening — a slow, inconsistent, and bias-prone process. This project automates resume classification to match candidates to the right job roles instantly.

#### Approach
- Parse and clean raw resume text
- Extract features using **TF-IDF Vectorization**
- Train a **multi-class classifier** to categorize resumes by job domain
- Evaluate with accuracy, precision, recall, and classification report

#### Pipeline

```
Raw Resume Text
      │
      ▼
┌────────────────────┐
│  Text Cleaning     │  ← Lowercase, remove punctuation/stopwords
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│  TF-IDF            │  ← Convert text to numerical feature vectors
│  Vectorization     │
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│  Classification    │  ← KNN / Logistic Regression / SVM
│  Model Training    │
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│  Evaluation &      │  ← Accuracy, Classification Report,
│  Results           │    Confusion Matrix
└────────────────────┘
```

#### Key Techniques
| Component | Details |
|---|---|
| **Feature Extraction** | TF-IDF Vectorizer |
| **Models Tried** | K-Nearest Neighbors, Logistic Regression, SVM |
| **Evaluation** | Accuracy Score, Classification Report |
| **Dataset** | Resume dataset with labeled job categories |

#### Business Impact
Reduces manual HR screening time significantly  
Standardizes candidate evaluation across all applicants  
Eliminates keyword-based bias from manual shortlisting  

---

## 2. Sales Forecasting

### *Predicting future revenue with time series machine learning*

#### Problem
Businesses need accurate sales predictions to manage inventory, allocate budgets, and plan campaigns. This project builds a regression-based forecasting model on historical sales data to predict future demand.

#### Approach
- Load and explore historical sales data with EDA
- Perform time-based feature engineering (month, quarter, lag features)
- Train **regression models** to predict future sales values
- Evaluate using MAE, RMSE, and R² Score

#### Pipeline

```
Historical Sales Data (CSV)
           │
           ▼
┌──────────────────────────┐
│  EDA & Trend Analysis    │  ← Seasonality, trends, outliers
└───────────┬──────────────┘
            │
            ▼
┌──────────────────────────┐
│  Feature Engineering     │  ← Month, quarter, lag features,
│                          │    rolling averages
└───────────┬──────────────┘
            │
            ▼
┌──────────────────────────┐
│  Model Training          │  ← Linear Regression, Random Forest,
│                          │    XGBoost Regressor
└───────────┬──────────────┘
            │
            ▼
┌──────────────────────────┐
│  Forecast & Evaluation   │  ← MAE, RMSE, R² Score,
│                          │    Actual vs Predicted plot
└──────────────────────────┘
```

#### Key Techniques
| Component | Details |
|---|---|
| **Feature Engineering** | Lag features, rolling means, time-based splits |
| **Models Tried** | Linear Regression, Random Forest, XGBoost |
| **Evaluation Metrics** | MAE, RMSE, R² Score |
| **Visualization** | Actual vs Predicted trend plots |

#### Business Impact
Enables data-driven inventory and budget planning  
Identifies seasonal patterns and demand spikes early  
Replaces gut-feel forecasting with quantified predictions  

---

## 3. Support Ticket Classification

### *Routing customer support tickets automatically with NLP*

#### Problem
Support teams receive thousands of tickets daily — manually categorizing and routing them wastes agent time and delays resolution. This project uses NLP to automatically classify support tickets by issue type so they can be instantly routed to the right team.

#### Approach
- Clean and preprocess raw ticket text
- Vectorize using **TF-IDF** or **Count Vectorizer**
- Train a **multi-class text classifier** across ticket categories
- Evaluate model performance with accuracy and F1-score

#### Pipeline

```
Raw Support Ticket Text
         │
         ▼
┌──────────────────────────┐
│  Text Preprocessing      │  ← Remove noise, stopwords,
│                          │    tokenize, lemmatize
└───────────┬──────────────┘
            │
            ▼
┌──────────────────────────┐
│  Vectorization           │  ← TF-IDF / CountVectorizer
└───────────┬──────────────┘
            │
            ▼
┌──────────────────────────┐
│  Multi-class             │  ← Naive Bayes, Logistic Regression,
│  Classification          │    Random Forest
└───────────┬──────────────┘
            │
            ▼
┌──────────────────────────┐
│  Evaluation              │  ← Accuracy, F1-Score,
│                          │    Confusion Matrix
└──────────────────────────┘
```

#### Key Techniques
| Component | Details |
|---|---|
| **Feature Extraction** | TF-IDF Vectorizer / CountVectorizer |
| **Models Tried** | Naive Bayes, Logistic Regression, Random Forest |
| **Evaluation** | Accuracy, F1-Score, Confusion Matrix |
| **Output** | Predicted ticket category label |

#### Business Impact
Automates ticket routing — zero manual triage needed  
Reduces average response and resolution time  
Scales effortlessly as ticket volume grows  

---

## Common Tech Stack

| Category | Tools |
|---|---|
| **Language** | Python 3.x |
| **Environment** | Jupyter Notebook / Google Colab |
| **Data Handling** | `pandas`, `numpy` |
| **ML Models** | `scikit-learn` |
| **NLP** | `nltk`, `re`, `TfidfVectorizer` |
| **Visualization** | `matplotlib`, `seaborn` |

---

## Getting Started

```bash
# Clone the repository
git clone https://github.com/jasvi05/Decision-Intelligence-ML.git
cd Decision-Intelligence-ML

# Install common dependencies
pip install pandas numpy scikit-learn matplotlib seaborn nltk

# Download NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"

# Navigate to any project and open its notebook
cd resume-screening-system
jupyter notebook
```

> Each subfolder is self-contained — open the `.ipynb` notebook inside each project folder to run it independently.

---

## Future Scope

| Project | Enhancement |
|---|---|
| Resume Screening | Add skill extraction, experience scoring, and job description matching |
| Sales Forecasting | Integrate LSTM / Prophet for deep time series modeling |
| Ticket Classification | Use BERT embeddings for context-aware classification |
| All Projects | Build a unified Streamlit dashboard for all 3 modules |

---

## Author

**Jasvi Desai** — [@jasvi05](https://github.com/jasvi05)

> *Decision Intelligence ML demonstrates how machine learning can be applied to real, high-impact business problems — replacing slow manual workflows with fast, scalable, and data-driven automation across HR, sales, and customer support.*

---

<div align="center">
⭐ If you found this project useful, consider starring the repository!
</div>
