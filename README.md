# Autism Spectrum Disorder Prediction

## 1. Introduction

This project aims to predict whether a patient is on the autism spectrum based on various features such as age, gender, ethnicity, and scores on the Autism Spectrum Quotient (AQ) 10 item screening tool. The prediction model could aid in early detection and intervention for individuals on the autism spectrum.

## 2. Dataset

The dataset used in this project consists of two files: `train.csv` and `test.csv`. The training set is used to train our machine learning models, while the test set is used to evaluate their performance.

### 2.1 Fields Description

The dataset contains the following fields:

- `ID`: ID of the patient
- `A1_Score` to `A10_Score`: Score based on Autism Spectrum Quotient (AQ) 10 item screening tool
- `age`: Age of the patient in years
- `gender`: Gender of the patient
- `ethnicity`: Ethnicity of the patient
- `jaundice`: Whether the patient had jaundice at the time of birth
- `autism`: Whether an immediate family member has been diagnosed with autism
- `contry_of_res`: Country of residence of the patient
- `used_app_before`: Whether the patient has undergone a screening test before
- `result`: Score for AQ1-10 screening test
- `age_desc`: Age of the patient
- `relation`: Relation of patient who completed the test
- `Class/ASD`: Classified result as 0 or 1. Here 0 represents No and 1 represents Yes. This is the target column, and during submission submit the values as 0 or 1 only.

The dataset seems to originate from a screening process for ASD. It might have been collected through a screening app, as suggested by the `used_app_before` field. This dataset could be useful for developing a model to predict whether an individual falls on the autism spectrum based on their screening results and other characteristics. Such a model could potentially aid in the early detection and diagnosis of ASD.

## 3. Prediction Goal

In this project, we aim to predict whether an individual falls on the Autism Spectrum Disorder (ASD) based on their screening results and other characteristics. The target variable is `Class/ASD`, which indicates whether the individual is on the autism spectrum.

## 3.1 Use in Practice

The prediction model developed in this project could be used in several ways:

- **Early Detection and Diagnosis**: The model could aid in the early detection and diagnosis of ASD. By predicting the likelihood of an individual being on the autism spectrum based on their screening results, it could help identify individuals who may benefit from a more thorough diagnostic evaluation.

- **Screening Tool**: The model could be integrated into a screening app or tool. Users could input their information and receive an immediate prediction. This could make the screening process more efficient and accessible.

- **Research and Development**: The model could also be used in research settings to better understand the factors that contribute to ASD. The feature importances learned by the model could provide insights into the relationships between different characteristics and ASD.

## 4. Process Overview

This section outlines the iterative approach taken during the project, including any pivots or adjustments made:

- **Data Loading and Preprocessing**: The initial step involved loading the data from CSV files and performing basic preprocessing tasks. This included dropping unnecessary columns, replacing string values with numerical values for easier processing, and converting the 'age' column to integer type.

- **Feature Engineering**: Certain features were label encoded to convert categorical variables into numerical variables. Interaction terms were also created for the 'age' and 'result' columns using PolynomialFeatures.

- **Handling Missing Values**: The KNNImputer was used to fill any missing values in the dataset based on the 5 nearest neighbors.

- **Data Splitting**: The dataset was split into features (X) and the target variable (y). This was then split into training and validation sets.

- **Data Scaling**: The features were scaled using StandardScaler to ensure that all features have a mean of 0 and a standard deviation of 1.

- **Handling Imbalanced Data**: SMOTE (Synthetic Minority Over-sampling Technique) was used to oversample the minority class in the target variable.

- **Model Building and Evaluation**: Four different models (Random Forest, XGBoost, Gradient Boosting, and Decision Tree) were trained using GridSearchCV for hyperparameter tuning. The models were fit on the oversampled training data. The best parameters for each model were printed, and the models were evaluated on the validation set. The accuracy, mean absolute error, classification report, and confusion matrix were printed for each model.

- **Final Model Training and Evaluation**: A final Random Forest classifier was trained on the last training set and evaluated on a separate validation set.

Throughout this process, various models were evaluated iteratively, with adjustments to feature engineering and hyperparameter tuning based on model performance. This iterative approach allowed for continuous improvement and refinement of the models.

## 5. Exploratory Data Analysis

In this project, we performed an extensive exploratory data analysis to understand the distribution of each feature and the correlation between features. Here are the key points:

- **X and Y Variables**: Our X variables include all the features except for 'Class/ASD'. Our Y variable is 'Class/ASD'. This is a classification problem.

- **Observations**: The dataset contains several observations (rows). Each observation represents a patient.

- **Features to Observations Ratio**: The ratio of features to observations was suitable for the machine learning models used.

- **Distribution of Each Feature**: We focused on features that are imbalanced or may present a problem when training a model. For example, we used SMOTE to handle the imbalance in the target variable 'Class/ASD'.

- **Distribution of Y**: The target variable 'Class/ASD' is binary and was imbalanced. We addressed this using SMOTE.

- **Correlation**: We created a correlation matrix to understand if some features are strongly correlated. This information can be useful for feature selection and engineering. The heatmap shows the correlation between different variables in your dataset. The variables include "A1_Score" through "A10_Score," "age," "gender," "ethnicity," "jaundice," "austim," "country_of_res," and others. Each square in the heatmap represents the correlation between two variables, with the color indicating the strength and direction of the correlation. Most squares are in shades of blue, indicating varying degrees of positive correlation, while some squares are in red, indicating negative correlations.

![alt text](<corr matrix-1.png>)

- **Feature Importance**: Feature importance was evaluated using the feature importances provided by the Random Forest model.

- **Feature Selection**: We used all features for X after preprocessing. The importance of each feature was evaluated using the feature importances provided by the Random Forest model.

- **Feature Engineering**: Certain features were label encoded to convert categorical variables into numerical variables. Interaction terms were also created for the 'age' and 'result' columns using PolynomialFeatures.

- **Score Distributions**: We plotted a series of bar graphs, each representing score distributions for different AI scores from AI_Score Distribution to AI9_Score Distribution. The x-axis of each graph represents score ranges, while the y-axis represents the frequency or count of those scores. The bars in each graph are colored in shades of green, with darker greens indicating higher scores or counts.

![alt text](output-5.png)

## 6. Data Preprocessing

In this project, we performed several data preprocessing steps to prepare the data for model fitting:

- **Handling Missing Values**: We used KNNImputer to fill any missing values in the dataset based on the 5 nearest neighbors.

- **Encoding Categorical Variables**: Certain features were label encoded to convert categorical variables into numerical variables.

- **Scaling Numerical Variables**: The features were scaled using StandardScaler to ensure that all features have a mean of 0 and a standard deviation of 1.

- **Creating Interaction Terms**: Interaction terms were created for the 'age' and 'result' columns using PolynomialFeatures.

## 7. Model Fitting and Evaluation

- **Train/Test Splitting**: The dataset was split into features (X) and the target variable (y). This was then split into training and validation sets. The split was performed using a standard 80/20 train/test split.

- **Data Leakage**: As the data does not contain any time series or future information, the risk of data leakage is minimal in this case.

- **Model Selection**: We selected four models for this task: Random Forest, XGBoost, Gradient Boosting, and Decision Tree. These models were chosen due to their ability to handle both linear and non-linear relationships between features and the target variable. They also have built-in methods for handling overfitting, which makes them suitable for this task.

- **Hyperparameter Selection**: For each model, we used GridSearchCV for hyperparameter tuning. This method performs a grid search over specified parameter values for an estimator. The parameters of the estimator that gave the best score on the left-out data were chosen.

- **Cross-Validation**: We used cross-validation during the hyperparameter tuning process to get a more robust estimate of the model performance. This was particularly important due to the imbalanced nature of our target variable.

## 8. Handling Imbalanced Data

We used Synthetic Minority Over-sampling Technique (SMOTE) to handle the imbalance in the target variable. This technique generates synthetic samples of the minority class, helping to improve the performance of the model on the minority class.

Before applying SMOTE, the target variable was imbalanced. This was evident in the "Class Distribution Before SMOTE" bar chart, which showed a higher count for class '0' compared to class '1'.

![alt text](output-1-1.png)

## 9. Model Evaluation and Validation

In this project, we used several metrics to evaluate the performance of our models:

- **Accuracy**: This was the primary metric used for model evaluation. Accuracy measures the proportion of correct predictions made by the model. It is a useful metric when the classes in the target variable are balanced.

- **Mean Absolute Error (MAE)**: This metric measures the average magnitude of the errors in a set of predictions, without considering their direction. It's a useful metric to quantify the prediction error of the model.

- **Classification Report**: This report includes the precision, recall, F1-score, and support for each class. These metrics provide a more comprehensive view of the model's performance.

- **Confusion Matrix**: This matrix provides a summary of the correct and incorrect predictions broken down by each category. It provides insights into the types of errors being made by the model.

### 9.1. Model Performance

The performance of the models was evaluated under three scenarios: without SMOTE, with SMOTE, and with SMOTE and sampling strategy. The mean accuracy of each model under each scenario was as follows:

- **Without SMOTE**: 
    - Random Forest: 0.8387499999999999
    - Gradient Boosting: 0.8474999999999999
    - Decision Tree: 0.845
    - XGBoost: 0.8575000000000002

    The confusion matrices for each model were as follows:

    - Random Forest:
        ```
        [[114  13]
         [ 12  21]]
        ```
    - Gradient Boosting:
        ```
        [[117  10]
         [ 11  22]]
        ```
    - Decision Tree:
        ```
        [[115  12]
         [ 14  19]]
        ```
    - XGBoost:
        ```
        [[116  11]
         [ 10  23]]
        ```

![alt text](output-2-1.png)

- **With SMOTE**: 
    - Random Forest: 0.9053400735294119
    - Gradient Boosting: 0.9030085784313726
    - Decision Tree: 0.8638848039215686
    - XGBoost: 0.9100306372549021

    The confusion matrices for each model were as follows:

    - Random Forest:
        ```
        [[109  18]
         [  9 119]]
        ```
    - Gradient Boosting:
        ```
        [[111  16]
         [  8 120]]
        ```
    - Decision Tree:
        ```
        [[102  25]
         [  9 119]]
        ```
    - XGBoost:
        ```
        [[109  18]
         [  7 121]]
        ```

![alt text](output-3-1.png)

- **With SMOTE and Sampling Strategy**: 
    - Random Forest: 0.8937771281778722
    - Gradient Boosting: 0.8843741695455754
    - Decision Tree: 0.8468110550093011
    - XGBoost: 0.8853219948622553

    The confusion matrices for each model were as follows:

    - Random Forest:
        ```
        [[111  16]
         [ 11  74]]
        ```
    - Gradient Boosting:
        ```
        [[110  17]
         [ 13  72]]
        ```
    - Decision Tree:
        ```
        [[111  16]
         [ 15  70]]
        ```
    - XGBoost:
        ```
        [[111  16]
         [ 12  73]]
        ```

![alt text](output-4-1.png)

The bar charts titled "Comparison of Model Performance - Without SMOTE", "Comparison of Model Performance - With SMOTE", and "Comparison of Model Performance - With SMOTE and Sampling Strategy" visually represent these results.

## 10. Production

- **Model Deployment**: The models trained in this project (Random Forest, XGBoost, Gradient Boosting, and Decision Tree) are suitable for deployment in a production environment. The models can be saved using libraries like `pickle` or `joblib`, and then loaded in the production environment to make predictions.

- **Model Use**: The models expect input in the same format as the training data. Any new data must go through the same preprocessing steps (handling missing values, encoding categorical variables, scaling numerical variables, etc.) before being passed to the model for prediction.

- **Precautions**: Here are some precautions about the use of these models:
    - The models were trained on a specific dataset. If the distribution of the new data is significantly different from the training data, the models may not perform well.
    - The models assume that the input data has the same structure as the training data. If any features are missing or additional features are included, the models may not work correctly.
    - The models were trained with a specific random state for reproducibility. Changing the random state may lead to different results.
    - The models use certain hyperparameters that were tuned for this specific problem. If the problem changes, the hyperparameters may need to be retuned.

## 11. Going Further

While the models in this project have shown promising results, there are several ways that they could potentially be improved:

- **More Data**: Collecting more data could help improve the performance of the models. This could include more observations or more features. More data can help the models learn more complex patterns and generalize better to unseen data.

- **Additional Features**: Incorporating additional features could also improve model performance. These could be new variables that provide additional information about the observations, or engineered features created from existing variables.

- **Feature Selection**: Applying feature selection techniques could help identify the most important features and remove irrelevant or redundant features. This could improve model interpretability and potentially also model performance.

- **Model Tuning**: Further tuning the models' hyperparameters could potentially improve their performance. This could involve a more extensive grid search, a random search, or more advanced methods like Bayesian optimization.

- **Model Stacking or Ensemble Methods**: Combining the predictions of multiple models could potentially improve performance. Techniques like stacking, bagging, or boosting could be used.

- **Data Augmentation**: For imbalanced datasets, more sophisticated techniques for oversampling the minority class or undersampling the majority class could be used. This could help improve the models' performance on the minority class.

## 12. Acknowledgements

We would like to thank the providers of the dataset for making this project possible. We would also like to thank our peers for their valuable feedback and suggestions throughout the project.