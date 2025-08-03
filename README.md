# Titanic Kaggle Competition – Machine Learning Project

This project is my submission to the [Titanic - Machine Learning from Disaster](https://www.kaggle.com/c/titanic) competition on Kaggle. It explores various preprocessing strategies, feature selection techniques, and classification models to predict passenger survival.

## Models Compared

- K-Nearest Neighbors
- Logistic Regression (L1 and L2 regularization)
- Decision Tree Classifier
- Bagging on Decision Trees
- Random Forest Classifier
- Gradient Boosting Classifier
- AdaBoost Classifier

## Preprocessing Combinations

Each model was tested across combinations of:

- **Imputation:** KNN imputer, KNN with missingness indicators
- **Categorical Encoding:** One-hot encoding (`get_dummies`), Label encoding
- **Scaling:** No scaling, Normalization, Standardization
- **Feature Selection:** With or without model-based selector
- **Hyperparameter Tuning:** Grid/Random search enabled/disabled

These combinations were run using parameter grids, and results were saved to CSV files for later analysis.

## Result Analysis

- All results were aggregated into a single Excel file with using Python.
- A Power BI report (`analysis.pbix`) was created to compare model and preprocessing performance.
- The best performing configuration used **Gradient Boosting** with tuned hyperparameters and specific preprocessing steps.
- Final submission achieved **0.78947** on Kaggle public leaderboard.

## AWS EC2 Deployment

The model comparison was run on an AWS EC2 `t2.large` spot instance:
- Total runtime: ~1 hour
- Cost: ~$0.10

See `/aws_ec2/` for requirements, data setup, and script version of the pipeline.

## Folder Overview

- `data/`: Training and test datasets
- `notebooks/`: Jupyter Notebooks for preprocessing, modeling, and final Kaggle submission
- `aws_ec2/`: Setup for running the pipeline on an EC2 instance
- `analysis/`: Script for result aggregation and Power BI analysis
- `output/`: Raw CSV results from model runs
- `requirements.txt`: Python dependencies

## How to Run Locally

**Clone the repository:**

- git clone https://github.com/DavidPoledne/kaggle_titanic.git
- cd kaggle_titanic

**Install dependencies:**

- pip install -r requirements.txt

**Run Jupyter Notebooks:**

- jupyter notebook notebooks/preprocessing.ipynb
- jupyter notebook notebooks/model_comparison.ipynb
