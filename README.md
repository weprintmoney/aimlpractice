# Welcome to the Classical Machine Learning Practice Repository

![AIML](https://img.shields.io/badge/AIML-Practice-brightgreen.svg)
![License](https://img.shields.io/github/license/weprintmoney/aimlpractice.svg)

This repository contains a set of practice assignments for Artificial Intelligence and Machine Learning. Each assignment covers a specific topic in AI/ML and is accompanied by a dataset for analysis.

## Table of Contents

- [Fundamentals of AIML](#fundamentals-of-aiml)
- [Supervised Learning: Regression](#supervised-learning-regression)
- [Supervised Learning: Classification](#supervised-learning-classification)
- [Ensemble Techniques](#ensemble-techniques)
- [Feature Selection, Model Selection, & Tuning](#feature-selection-model-selection--tuning)
- [Unsupervised Learning](#unsupervised-learning)
- [Deep Neural Networks](#deep-neural-networks)
- [Computer Vision](#computer-vision)
- [Natural Language Processing](#natural-language-processing)
- [How to Use](#how-to-use)

## Fundamentals of AIML
**Goals:**
Execute preliminary data analysis by exploring the dataset and come up with some basic observations about the data. Build customer profile to help capitalize based on it. Extract actionable insights that drive the sales of the business.

**Skills & Tools Covered:**
- Pandas
- NumPy
- Visualisation techniques
- Exploratory Data Analysis (EDA)

**Notebook & Dataset:**
- [Cardio Good Fitness Project](https://github.com/weprintmoney/aimlpractice/blob/main/Cardio%20Good%20Fitness%20Project.ipynb)
  - Dataset: [CardioGoodFitness.csv](https://drive.google.com/file/d/1w0RZRoBmiIfO5HeZhghZeG9yRRRwP7Px/view?usp=sharing)

## Supervised Learning: Regression
**Goals:**
Create a pricing model that can effectively predict the price of used cars to help the business in devising profitable strategies using differential pricing.

**Skills & Tools Covered:**
- EDA
- Linear regression
- Linear regression assumptions
- Business insights and suggestions

**Notebook & Dataset:**
- [Cars4U Project](https://github.com/weprintmoney/aimlpractice/blob/main/Cars4U%20Project.ipynb)
  - Dataset: [used_cars_data.csv](https://drive.google.com/file/d/18i_OKMqt5G33NjabFfCv2YKJHi5xvjya/view?usp=share_link)

## Supervised Learning: Classification
**Goals:**
Build a model to help the marketing department identify potential customers who have a higher probability of purchasing the loan.

**Skills & Tools Covered:**
- EDA
- Data Pre-processing
- Logistic regression
- Finding optimal threshold using AUC-ROC curve
- Decision trees
- Pruning

**Notebook & Dataset:**
- [Personal Loan Campaign](https://github.com/weprintmoney/aimlpractice/blob/main/Personal%20Loan%20Campaign.ipynb)
  - Dataset: [Loan_Modelling.csv](https://drive.google.com/file/d/1GkKw2gnlevH7oc3GKktZuo2BZpDBPoFv/view?usp=share_link)

## Ensemble Techniques
**Goals:**
Analyze the customers' information and build a model to predict the potential customers who will purchase a newly introduced package.

**Skills & Tools Covered:**
- EDA
- Data Preprocessing
- Customer Profiling
- Bagging Classifiers - Bagging and Random Forest
- Boosting Classifier - AdaBoost
- Gradient Boosting
- XGBoost
- Stacking Classifier
- Hyperparameter Tuning using GridSearchCV 
- Business Recommendations

**Notebook & Dataset:**
- [Travel Package Purchase Prediction](https://github.com/weprintmoney/aimlpractice/blob/main/Travel%20Package%20Purchase%20Prediction.ipynb)
  - Dataset: [Tourism.xlsx](https://docs.google.com/spreadsheets/d/1t0L2Ly37nkMq2UJqCwfCh5dO3zCFGJs-/edit?usp=share_link&ouid=105207491410102480746&rtpof=true&sd=true)

## Feature Selection, Model Selection, & Tuning
**Goals:**
Predict if a customer will leave the credit card services or not and the reasons why.

**Skills & Tools Covered:**
- Cross validation
- Regularization
- Pipelines and Hyperparameter tuning
- Up and Down Sampling

**Notebook & Dataset:**
- [Credit Card Users Churn Prediction](https://github.com/weprintmoney/aimlpractice/blob/main/Credit%20Card%20Users%20Churn%20Prediction.ipynb)
  - Dataset: [BankChurners.csv](https://drive.google.com/file/d/1o7DWPHcmaNsE4MAy6FHLeBe_Ej4I5KQY/view?usp=share_link)

## Unsupervised Learning
**Goals:**
Identify different segments in the existing customer base based on their spending patterns as well as past interaction with the bank and provide recommendations to the bank on how to better market to and service these customers.

**Skills & Tools Covered:**
- EDA
- Clustering (K-means and Hierarchical)
- Cluster Profiling

**Notebook & Dataset:**
- [AllLife Bank Customer Segmentation](https://github.com/weprintmoney/aimlpractice/blob/main/AllLife%20Bank%20Customer%20Segmentation.ipynb)
  - Dataset: [Credit Card Customer Data.xlsx](https://docs.google.com/spreadsheets/d/1RXnmettXZ1E4wNtz8aHYdULhrsXWzPdt/edit?usp=share_link&ouid=105207491410102480746&rtpof=true&sd=true)

## Deep Neural Networks
**Goals:**
Help the operations team identify the customers that are more likely to churn by building an artificial Neural Network from scratch.

**Skills & Tools Covered:**
- Tensorflow
- Keras
- ANN
- Google colab

**Notebook & Dataset:**
- [Bank Churn Prediction](https://github.com/weprintmoney/aimlpractice/blob/main/Bank%20Churn%20Prediction.ipynb)
  - Dataset: [churn.csv](https://drive.google.com/file/d/1LDsb1fzbMq-jK4BHVKD8SNEsPSZmWEhX/view?usp=share_link)

## Computer Vision
**Goals:**
Identify the plant seedlings species from 12 different species using a convolutional neural network

**Skills & Tools Covered:**
- Keras
- CNN
- Working With Images
- Computer Vision

**Notebook & Dataset:**
- [Plants Seedling Classification](https://github.com/weprintmoney/aimlpractice/blob/main/Plants%20Seedling%20Classification.ipynb)
  - Dataset: [labels.csv](https://drive.google.com/file/d/1-EYgmxPDmwIBA6lvSCXTN6VeUx0Hys_A/view?usp=share_link)
  - Dataset: [images.npy](https://drive.google.com/file/d/1d6goNSozkTvoAYIXkIKKDg6A8sZp51F1/view?usp=share_link)

## Natural Language Processing
**Goals:**
Identify the sentiment from a tweet to understand an airlines' customer satisfaction

**Skills & Tools Covered:**
- Working with text
- Vectorization(Count vectorizer & tf-idf vectorizer)
- Sentiment analysis
- Parameter tuning
- Confusion matrix based model evaluation

**Notebook & Dataset:**
- [Twitter US Airline Sentiment](https://github.com/weprintmoney/aimlpractice/blob/main/Twitter%20US%20Airline%20Sentiment.ipynb)
  - Dataset: [Tweets.csv](https://drive.google.com/file/d/1ti9wfyVMgbojs39LGh6UhwubIJRey487/view?usp=share_link)

## How to Use

To use the datasets in this repository, please follow the steps below:

1. Copy the dataset to your own Google Drive account by clicking on the dataset link in the assignment description above.
2. Open the dataset in your Google Drive account and click on the "Add shortcut to Drive" button in the upper right corner.
3. In the popup window, select the folder in your Google Drive where you want to add the dataset shortcut, and then click on the "Add shortcut" button.
4. In your local machine or Google Colab notebook, mount your Google Drive using the following code:

```python
from google.colab import drive
drive.mount('/content/drive')
```

5. Access the dataset using the file path of the dataset shortcut in your Google Drive, like this:

```python
import pandas as pd
df = pd.read_csv('/content/drive/My Drive/<folder>/<dataset>')
```

Replace `<folder>` with the name of the folder where you added the dataset shortcut in step 3, and `<dataset>` with the name of the dataset file.

Thank you for checking out my repository. I hope that these assignments will help you gain more knowledge and expertise in AI/ML. If you have any suggestions or feedback, please feel free to contribute or contact me.
