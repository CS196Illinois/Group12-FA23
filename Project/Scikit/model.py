# Import Statements
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, classification_report, precision_score)
import matplotlib.pyplot as plt
# We will want to use k-fold for the final model optimally instead of train_test_split


# Global Variables
data_location = 'data.csv'
game_dataframe = pd.read_csv(data_location)
game_dataframe = game_dataframe.drop(columns=['date', 'Team A', 'TeamB', 'A_score', 'B_score', 'A_minus_B'])

game_dataframe = pd.get_dummies(game_dataframe, drop_first=True)

# Test Train Split
X_train, X_test, y_train, y_test = train_test_split(
    game_dataframe.drop(columns=['A_result_Tie', 'A_result_Win']), 
    game_dataframe['A_result_Win'],
    test_size = 0.2,
    random_state = 0
)

# Train the model using the training data
LogReg = LogisticRegression(solver='liblinear')
LogReg.fit(X_train, y_train)

y_pred = LogReg.predict(X_test)

confusion_matrix(y_test, y_pred)

print(classification_report(y_test, y_pred))

print("Precision score: {}".format(precision_score(y_test, y_pred)))