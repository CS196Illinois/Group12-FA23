# Import Statements
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split 
# We will want to use k-fold for the final model optimally instead of train_test_split


# Global Variables
data_location = 'data.csv'
game_dataframe = pd.read_csv(data_location)
game_dataframe = game_dataframe.drop(columns=['date', 'Team A', 'TeamB', 'A_minus_B'])

game_dataframe = pd.get_dummies(game_dataframe, drop_first=True)

# Test Train Split
X_train, X_test, y_train, y_test = train_test_split(
    game_dataframe.drop(columns=['A_result_Tie', 'A_result_Win']), 
    game_dataframe['A_result_Win']
)

# Train the model using the training data
LogReg = LogisticRegression(solver='lbfgs')
LogReg.fit(X_train, y_train)

print(LogReg.predict(X_test))