# Breast Cancer Data Analysis and Prediction
This project focuses on analyzing a breast cancer dataset and building a machine learning model to predict the "6th Stage" of cancer based on various patient attributes. The project is divided into two main Jupyter Notebooks: one for exploratory data analysis (EDA) and data cleaning, and another for building and evaluating the predictive model.

## Table of Contents
### Project Overview

#### Dataset

Files in this Repository

Key Features and Analysis

Machine Learning Model

Dependencies

## How to Use

Contributing

License

## Project Overview
The goal of this project is to leverage a comprehensive breast cancer dataset to:

Perform in-depth exploratory data analysis to understand the characteristics and distributions of the data.

Clean and preprocess the raw data, handling categorical variables and addressing class imbalance.

Develop a machine learning model capable of predicting the 6th stage of breast cancer, which can potentially aid in early diagnosis or prognosis.

Dataset
The primary dataset used in this project is Breast_Cancer.csv. It contains various attributes related to breast cancer patients, including:

Age

Race

Marital Status

T Stage

N Stage

6th Stage (Target variable)

differentiate

Grade

A Stage

Tumor Size

Estrogen Status

Progesterone Status

Regional Node Examined

Reginol Node Positive

Survival Months

Status

The dataset is initially loaded and explored in data cancer.ipynb, and a cleaned version (step1_cleanCancer.csv) is generated for the modeling phase.

## Files in this Repository
data cancer.ipynb: This notebook covers the initial data loading, exploratory data analysis (EDA), and data cleaning steps. It includes checks for missing values, duplicates, data types, unique values, descriptive statistics, and a scatter plot to visualize the relationship between 'Age' and '6th Stage'. The cleaned data is then saved as step1_cleanCancer.csv.

datacancer_model.ipynb: This notebook focuses on building and evaluating the machine learning model. It loads the cleaned data, performs further preprocessing (one-hot encoding for categorical variables), addresses class imbalance using SMOTE (Synthetic Minority Over-sampling Technique), trains a Random Forest Classifier, evaluates its accuracy, and saves the trained model as breast_cancer_model.pkl for future predictions.

## Key Features and Analysis
Data Exploration and Cleaning (data cancer.ipynb)
Initial Data Overview: Understanding the dataset's dimensions, checking for null values, and identifying duplicate entries.

Data Type and Uniqueness Analysis: Examining the data types of each column and the number of unique values to inform preprocessing strategies.

Descriptive Statistics: Summarizing numerical features to understand their central tendency, dispersion, and shape.

Age vs. Cancer Stage Visualization: A scatter plot is used to visually inspect the relationship between patient age and the 6th cancer stage, revealing insights such as:

No direct linear trend between age and cancer stage.

Presence of advanced stages (IIIB, IIIC) in older individuals, suggesting a potential higher likelihood of late-stage diagnosis.

Occurrence of aggressive cancer stages (IIIC) even in younger patients.

Machine Learning Model (datacancer_model.ipynb)
Target Variable Distribution: Analysis of the '6th Stage' distribution to identify and address class imbalance.

Feature Engineering: Categorical features are converted into numerical format using One-Hot Encoding.

Class Imbalance Handling: SMOTE is applied to the training data to oversample minority classes, ensuring the model is not biased towards the majority class.

Model Training: A Random Forest Classifier is trained on the preprocessed and balanced dataset.

Model Evaluation: The model's performance is assessed using accuracy score on the test set.

Model Persistence: The trained Random Forest model is saved using joblib (and pickle in an earlier cell) to breast_cancer_model.pkl, allowing for easy loading and future predictions without retraining.

Dependencies
To run these notebooks, you'll need the following Python libraries:

pandas

numpy

matplotlib

seaborn

scikit-learn

imblearn (specifically imbalanced-learn for SMOTE)

joblib

You can install these dependencies using pip:

pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn joblib

How to Use
Follow these steps to set up and run the project:

Clone the Repository:
If you have Git installed, you can clone this repository to your local machine:

git clone https://github.com/YourGitHubUsername/YourRepositoryName.git
cd YourRepositoryName

(Replace YourGitHubUsername and YourRepositoryName with your actual GitHub username and the name you give your repository.)

# Download the Dataset:
Ensure you have the Breast_Cancer.csv file in the same directory as your notebooks. If it's not included in the repository, you'll need to obtain it from its original source and place it there.

Install Dependencies:
Navigate to the project directory in your terminal and install the required libraries:

pip install -r requirements.txt

(Note: You'll need to create a requirements.txt file first. See the next section for how to do this.)

Run the Jupyter Notebooks:
Launch Jupyter Notebook:

jupyter notebook

This will open a browser window with the Jupyter interface. From there, you can open and run the notebooks in the following order:

data cancer.ipynb: Run all cells to perform EDA and generate step1_cleanCancer.csv.

datacancer_model.ipynb: Run all cells to train the model, evaluate it, and save breast_cancer_model.pkl.

Contributing
Contributions are welcome! If you have suggestions for improvements, new features, or bug fixes, please feel free to:

Fork the repository.

Create a new branch (git checkout -b feature/YourFeature).

Make your changes.

Commit your changes (git commit -m 'Add some feature').

Push to the branch (git push origin feature/YourFeature).

Open a Pull Request.

License
This project is open-sourced under the MIT License. See the LICENSE file for more details.
