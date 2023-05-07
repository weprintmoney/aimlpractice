# Welcome to the AI ML Practice Repository

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

- Cardio Good Fitness Project
  - Dataset: [CardioGoodFitness.csv](https://drive.google.com/file/d/1w0RZRoBmiIfO5HeZhghZeG9yRRRwP7Px/view?usp=sharing)

## Supervised Learning: Regression

- Cars4U Project
  - Dataset: [used_cars_data.csv](https://drive.google.com/file/d/18i_OKMqt5G33NjabFfCv2YKJHi5xvjya/view?usp=share_link)

## Supervised Learning: Classification

- Personal Loan Campaign
  - Dataset: [Loan_Modelling.csv](https://drive.google.com/file/d/1GkKw2gnlevH7oc3GKktZuo2BZpDBPoFv/view?usp=share_link)

## Ensemble Techniques

- Travel Package Purchase Prediction
  - Dataset: [Tourism.xlsx](https://docs.google.com/spreadsheets/d/1t0L2Ly37nkMq2UJqCwfCh5dO3zCFGJs-/edit?usp=share_link&ouid=105207491410102480746&rtpof=true&sd=true)

## Feature Selection, Model Selection, & Tuning

- Credit Card Users Churn Prediction
  - Dataset: [BankChurners.csv](https://drive.google.com/file/d/1o7DWPHcmaNsE4MAy6FHLeBe_Ej4I5KQY/view?usp=share_link)

## Unsupervised Learning

- AllLife Bank Customer Segmentation
  - Dataset: [Credit Card Customer Data.xlsx](https://docs.google.com/spreadsheets/d/1RXnmettXZ1E4wNtz8aHYdULhrsXWzPdt/edit?usp=share_link&ouid=105207491410102480746&rtpof=true&sd=true)

## Deep Neural Networks

- Bank Churn Prediction
  - Dataset: [churn.csv](https://drive.google.com/file/d/1LDsb1fzbMq-jK4BHVKD8SNEsPSZmWEhX/view?usp=share_link)

## Computer Vision

- Plants Seedling Classification
  - Dataset: [labels.csv](https://drive.google.com/file/d/1-EYgmxPDmwIBA6lvSCXTN6VeUx0Hys_A/view?usp=share_link)
  - Dataset: [images.npy](https://drive.google.com/file/d/1d6goNSozkTvoAYIXkIKKDg6A8sZp51F1/view?usp=share_link)

## Natural Language Processing

- Twitter US Airline Sentiment
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
