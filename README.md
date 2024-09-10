# Gender Tracking with App Usage

Project by [Ranit Bhowmick](https://linktr.ee/ranitbhowmick) and [Sayanti Chatterjee](https://linktr.ee/sayantichatterjee)

## Overview

This project aims to predict the gender of individuals based on their app usage patterns. The project leverages data collected through a custom survey to train machine learning models. By analyzing various app usage statistics, such as time spent on different categories of apps, we aim to determine the user's gender with high accuracy.

The project encompasses data collection, preprocessing, and model training, utilizing techniques such as one-hot encoding, outlier detection, and machine learning algorithms like Decision Trees and Random Forest.

## Survey Information

To gather data for this project, we conducted a survey titled "What's Your App Usage?". The survey was designed to capture detailed information about participants' app usage across various categories. The data collected is crucial for training our machine learning models, and we appreciate the participation of everyone who took the time to contribute.

### Survey Questions
The survey was structured to be clean and organized, with each question focusing on specific aspects of app usage. Participants could fill out the form anonymously, and the sections included:
- Basic demographic information
- App usage duration across various categories (e.g., Social, Gaming, Banking)
- Time of app usage during the day

You can view and participate in the survey [here](https://forms.gle/gQFGemdu8aciNnZp6).

## Data Description

The dataset used for this project was constructed from the survey responses. The raw data includes columns such as `Transportation Usage`, `Social Usage`, `Meet Usage`, and more, all of which represent the time spent on various app categories.

### Dataset Features
The main features in the dataset are as follows:
- **App Usage Duration:** Time spent on apps in different categories, represented in `HH:MM:SS` format.
- **App Usage Time:** Time of day when apps in different categories were used, also in `HH:MM` format.
- **Demographic Information:** Gender, Employment Status, Field of Work, and Date of Birth.

### Data Preprocessing
#### 1. Handling Missing Values
The dataset contained some missing values in the app usage duration columns. These missing values were filled with `'00:00:00'`, indicating no usage.

#### 2. Outlier Detection and Correction
We implemented a custom outlier detection and correction function to handle invalid time entries. For instance, any time value with hours exceeding 23 was corrected to `'00:00:00'`.

#### 3. Converting Time to Numerical Values
To make the time data suitable for machine learning models, we converted the `HH:MM:SS` and `HH:MM` formats into total seconds. This conversion allows the models to process and analyze the time data effectively.

#### 4. Date of Birth Conversion
The `Date of Birth` was converted into the number of days since birth, which provides a numerical representation of the participant's age.

#### 5. One-Hot Encoding
Categorical variables such as `Gender`, `Employment Status`, and `Field` were converted into numerical values using one-hot encoding. This step is crucial for feeding the data into machine learning algorithms.

## Machine Learning Models

### 1. Decision Tree Classifier
We initially employed a Decision Tree Classifier to predict gender based on app usage patterns. The model was trained using the preprocessed dataset, and it achieved an accuracy of around 82% on the test data.

```python
from sklearn import tree

model = tree.DecisionTreeClassifier()
model.fit(x_train, y_train)
accuracy = model.score(x_test, y_test)
print(f"Decision Tree Accuracy: {accuracy}")
```

### 2. Random Forest Classifier
To further improve the accuracy, we used a Random Forest Classifier with 15 estimators. The Random Forest model provided more robust results, achieving similar accuracy levels as the Decision Tree but with improved stability.

```python
from sklearn.ensemble import RandomForestClassifier as rf

model2 = rf(n_estimators=15)
model2.fit(x_train, y_train)
accuracy_rf = model2.score(x_test, y_test)
print(f"Random Forest Accuracy: {accuracy_rf}")
```

## Results

- The Random Forest model achieved an accuracy of **82%**, which is promising given the limited dataset size.
- The Decision Tree model also performed well, but Random Forest's ensemble approach provided more consistent results.
- The accuracy can be further improved with a larger and more diverse dataset, as well as by fine-tuning the hyperparameters of the models.

## How to Run the Project

### 1. Prerequisites
Make sure you have Python 3.x installed along with the required libraries:
```bash
pip install pandas numpy scikit-learn
```

### 2. Clone the Repository
```bash
git clone https://github.com/Kawai-Senpai/Info-Through-App-Usage.git
cd Info-Through-App-Usage
```

### 3. Run the Code
Ensure that you have the dataset (`app_usage.csv`) in the same directory and run the Jupyter notebook or the Python script provided.

### 4. Analyze the Results
After running the models, you can view the accuracy scores and model performance metrics. The models can be further fine-tuned to improve predictions.

## Future Work

- **Expand the Dataset:** Collect more survey responses to enhance the training dataset's size and diversity.
- **Feature Engineering:** Explore additional features that might improve model accuracy, such as app usage frequency or session length.
- **Model Optimization:** Experiment with other machine learning models, such as Gradient Boosting or Support Vector Machines, and fine-tune hyperparameters.
- **Deployment:** Consider deploying the model as a web service or integrating it into an app to provide real-time gender predictions based on app usage.

## Acknowledgments

We thank all participants who took the time to complete our survey. Your contributions have been invaluable to this project. Special thanks to Sayanti Chatterjee for her collaboration and support.

## Contact

For more information, feel free to reach out:
- **Ranit Bhowmick:** [Linktree](https://linktr.ee/ranitbhowmick)
- **Sayanti Chatterjee:** [Linktree](https://linktr.ee/sayantichatterjee)
