**Project Name: Diabetes Risk Prediction**

---

## Overview:

This project aims to predict the risk of diabetes in patients based on their medical history and demographic information. Two classification algorithms, namely Bayesian classifier and Decision Tree classifier, are employed for this task. The dataset contains features such as age, gender, body mass index (BMI), hypertension, heart disease, smoking history, HbA1c level, and blood glucose level. The class label indicating the presence or absence of diabetes is the last column in the provided dataset.

## Features:

1. **User Interface:**
   - Enables the user to select the percentage of data to be read from the input file.
   - Allows the user to choose the file for analysis.

2. **Data Processing:**
   - Divides the dataset into two subsets: Training Set (75% of data) and Testing Set (25% of data).
   - Size of each dataset is determined by user input.
   - Class Label column is identified as the last column in the chosen file.

3. **Algorithms:**
   - Bayesian Classifier: Utilized to build a predictive model based on the Training Set.
   - Decision Tree Classifier: Another algorithm employed for building a predictive model.

4. **Outputs:**
   - Accuracy of the model for both Bayesian and Decision Tree classifiers.
   - Class labels predicted by the classifiers for the data records provided by the user.

## Instructions:

1. **Input File Selection:**
   - Select the dataset file containing patient records for analysis.

2. **Data Splitting:**
   - Specify the size of the Training Set and Testing Set as per your requirement.

3. **Model Building:**
   - Apply Bayesian and Decision Tree algorithms to build classifier models using the Training Set.

4. **Model Evaluation:**
   - Apply the trained models on the Testing Set to calculate the accuracy of both classifiers.

5. **Result Comparison:**
   - Compare the performance results of the Bayesian and Decision Tree classifiers.

## Usage:

1. **Input File:**
   - Provide a CSV or text file containing patient data.

2. **Percentage Split:**
   - Enter the percentage of data to be used for analysis.

3. **Execute Program:**
   - Run the program using your preferred programming language.

## Example Images:

![Training and Testing Set Split](t1.png)

![Model Comparison](t2.png)



