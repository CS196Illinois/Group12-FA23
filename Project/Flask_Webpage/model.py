# Binomial Logistic Regression Ignoring Ties
# Main Model (Other ones are testing)

# Import Statements
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


# Global Variables
data_location = 'data.csv'
game_dataframe = pd.read_csv(data_location)
game_dataframe = game_dataframe.drop(columns=['date', 'Team A', 'TeamB', 'A_score', 'B_score', 'A_minus_B'])

game_dataframe = pd.get_dummies(game_dataframe, drop_first=True)

# Data
X = game_dataframe.drop(columns=['A_result_Tie', 'A_result_Win'])
y = game_dataframe['A_result_Win']

# Define Model
LogReg = make_pipeline(StandardScaler(), LogisticRegression(solver='lbfgs'))

# Train Model
LogReg.fit(X.values, y.values)

### Export Model Predictions
def predict(params):
    return LogReg.predict_proba([params])

print("Model Loaded")

''' TEST CODE
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
'''