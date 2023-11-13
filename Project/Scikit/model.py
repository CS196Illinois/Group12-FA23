# Import Statements
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, classification_report, precision_score)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import make_pipeline
from sklearn.metrics import precision_score


# We will want to use k-fold for the final model optimally instead of train_test_split


# Global Variables
data_location = 'data.csv'
game_dataframe = pd.read_csv(data_location)

game_dataframe = game_dataframe.drop(
    columns=['date', 'Team A', 'TeamB', 'A_score', 'B_score', 'A_minus_B'])


game_dataframe = pd.get_dummies(game_dataframe, drop_first=True)

# Test Train Split
X_train, X_test, y_train, y_test = train_test_split(


    game_dataframe.drop(columns=['A_result_Tie', 'A_result_Win']),

    game_dataframe['A_result_Win'],
    test_size=0.2,
    random_state=42
)

# Train the model using the training data

# LogReg = make_pipeline(StandardScaler(), LogisticRegression(solver='lbfgs'))
# LogReg.fit(X_train, y_train)

# y_pred = LogReg.predict(X_test)

# experiment
C_values = [0.001, 0.01, 0.1, 1, 10, 100]

penalties = ['11', '12']

best_model = None
best_precision = 0

for penalty in penalties:
    for C_value in C_values:
        LogReg = make_pipeline(
            StandardScaler(),
            LogisticRegression(solver='lbfgs', penalty=penalty, C=C_value)

        )
        LogReg.fit(X_train, y_train)

        y_pred = LogReg.predict(X_test)
        precision = precision_score(y_test, y_pred)

        if precision > best_precision:
            best_precision = precision
            best_model = LogReg


print("Best Model: " + best_model)
print("Best Precision Score: ", best_precision)

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
print(classification_report(y_test, y_pred))
print("precision score: {}".format(precision_score(y_test, y_pred)))

prob = LogReg.predict_proba(X_test)

counter = 0
actual = np.array(y_test).tolist()
for index, line in enumerate(prob):
    if counter > 100:
        break

    line_with_prob = np.append(
        line, ['Predicted Probability: {:.3f}'.format(line[1])])

    if y_pred[index]:
        line_with_prob = np.append(line_with_prob, ['Predicted: Win'])
    else:
        line_with_prob = np.append(line_with_prob, ['Predicted: Lose/Tie'])

    if actual[index]:
        line_with_prob = np.append(line_with_prob, ['Actual: Win'])
    else:
        line_with_prob = np.append(line_with_prob, ['Actual: Lose/Tie'])

    counter += 1

    print(line_with_prob)
