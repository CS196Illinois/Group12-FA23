# Binomial Logistic Regression Ignoring Ties
# Main Model (Other ones are testing)

# Import Statements
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (confusion_matrix, classification_report, precision_score)
import matplotlib.pyplot as plt
# We will want to use k-fold for the final model optimally instead of train_test_split


# Global Variables
data_location = 'data.csv'
game_dataframe = pd.read_csv(data_location)
game_dataframe = game_dataframe.drop(columns=['date', 'Team A', 'TeamB', 'A_score', 'B_score', 'A_minus_B'])

game_dataframe = pd.get_dummies(game_dataframe, drop_first=True)

# Data Splits
X = game_dataframe.drop(columns=['A_result_Tie', 'A_result_Win'])
y = game_dataframe['A_result_Win']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

# Define Model
# LogReg = LogisticRegression(solver='liblinear')
LogReg = make_pipeline(StandardScaler(), LogisticRegression(solver='lbfgs'))

# Train Model
LogReg.fit(X_train, y_train)

# Predict Test Data
y_pred = LogReg.predict(X_test)

# Analyze Accuracy
print(classification_report(y_test, y_pred))
print("Precision score: {}".format(precision_score(y_test, y_pred)))

prob = LogReg.predict_proba(X_test)

counter = 0
actual = y_test.tolist()
for index, line in enumerate(prob):
    if counter > 100: break

    if y_pred[index]:
        line = np.append(line, ['Predicted: Win'])
    else:
        line = np.append(line, ['Predicted: Lose/Tie'])

    if actual[index]:
        line = np.append(line, ['Actual: Win'])
    else:
        line = np.append(line, ['Actual: Lose/Tie'])

    counter += 1

    print(line)