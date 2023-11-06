# Multinomial Logistic Regression

# Import Statements
import pandas as pd
import numpy as np
from numpy import (mean, std)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (train_test_split, cross_val_score, RepeatedStratifiedKFold)
from sklearn.metrics import (classification_report, precision_score)
import matplotlib.pyplot as plt
# We will want to use k-fold for the final model optimally instead of train_test_split


# Global Variables
data_location = 'data.csv'
game_dataframe = pd.read_csv(data_location)
game_dataframe = game_dataframe.drop(columns=['date', 'Team A', 'TeamB', 'A_score', 'B_score', 'A_minus_B'])

# Test Train Split
X_train, X_test, y_train, y_test = train_test_split(
    game_dataframe.drop(columns=['A_result']), 
    game_dataframe['A_result'],
    test_size = 0.2,
    random_state = 0
)

X = game_dataframe.drop(columns=['A_result'])
y = game_dataframe['A_result']

# Test K-fold Cross Validation
LogReg = LogisticRegression(multi_class='multinomial', solver='lbfgs')
pipe = make_pipeline(StandardScaler(), LogisticRegression(multi_class='multinomial', solver='lbfgs'))
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(pipe, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
print('Mean Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)

print(classification_report(y_test, y_pred))

prob = pipe.predict_proba(X_test)

counter = 0
for index, line in enumerate(prob):
    if counter > 100: break

    if y_pred[index]:
        line = np.append(line, ['Predicted: Win'])
    else:
        line = np.append(line, ['Predicted: Lose/Tie'])

    counter += 1

    print(line)