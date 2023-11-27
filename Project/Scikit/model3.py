# Decision Tree for Probabilities

# Import Statements
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
# We will want to use k-fold for the final model optimally instead of train_test_split


# Global Variables
data_location = 'data.csv'
game_dataframe = pd.read_csv(data_location)
game_dataframe = game_dataframe.drop(columns=['date', 'Team A', 'TeamB', 'A_score', 'B_score', 'A_minus_B'])

game_dataframe = pd.get_dummies(game_dataframe, drop_first=True)

X = game_dataframe.drop(columns=['A_result_Win', 'A_result_Tie'])
y = game_dataframe['A_result_Win']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

clf = DecisionTreeClassifier(criterion='entropy', max_depth=7)

clf = clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

prob = clf.predict_proba(X_test)

for index, line in enumerate(prob):
    if y_pred[index]:
        line = np.append(line, ['Predicted: Win'])
    else:
        line = np.append(line, ['Predicted: Lose/Tie'])

    print(line)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

print(y_test)