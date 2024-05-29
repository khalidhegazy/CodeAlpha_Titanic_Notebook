Titanic Survival Prediction
This repository contains a Jupyter Notebook for predicting the survival of passengers on the Titanic using three different machine learning algorithms: Logistic Regression, Random Forest Classifier, and K-Nearest Neighbors (KNN) Classifier.

Overview
The Titanic dataset is a famous dataset used in machine learning and statistics. The goal is to build a predictive model that can determine whether a passenger survived the Titanic disaster based on certain features such as age, sex, and class.

Dataset
The dataset used in this notebook is the Titanic dataset provided by Kaggle. It includes "Titanic-Dataset.csv" file .

Installation
numpy
pandas
scikit-learn
matplotlib
seaborn

Usage
Clone the repository:
git clone https://github.com/khalidhegazy/Titanic-Notebook

Models
The notebook implements three different machine learning models to predict the survival of Titanic passengers.

Logistic Regression
Logistic Regression is a linear model used for binary classification. It models the probability that a given input point belongs to a certain class.

Random Forest Classifier
Random Forest is an ensemble method that fits multiple decision trees on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.

KNN Classifier
K-Nearest Neighbors (KNN) is a non-parametric method used for classification. The input consists of the k closest training examples in the feature space, and the output is a class membership.

Results
Logistic Regression achieved a respectable accuracy, indicating its effectiveness for binary classification tasks with this dataset.
Random Forest Classifier showed improved performance, benefiting from its ability to handle complex relationships within the data.
GridSearchCV further enhanced the Random Forest model's accuracy, demonstrating the value of hyperparameter tuning.
K-Nearest Neighbors provided a comparative baseline, showcasing different algorithmic strengths and weaknesses.
