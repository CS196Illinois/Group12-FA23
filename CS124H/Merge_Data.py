# To add world rank and points at that game to the extracted data used for prediction.
import pandas as pd
import numpy as np

rank = pd.read_csv("averaged.csv", index_col=0)
results = pd.read_csv("Extracted_Data.csv")
rank = pd.DataFrame(rank)
results = pd.DataFrame(results)

results["A_rank"] = "-1"
results["A_points"] = "-1"
results["B_rank"] = "-1"
results["B_points"] = "-1"

dict = {'Team a': [],
        'Team b': [],
        'year': [],
        'A status': [],
        'B status': []
        }
notFound = pd.DataFrame(dict)

a = False
b = False
for index, row in results.iterrows():
    a = False
    b = False
    for i, num in rank.iterrows():
        if row["date"] < 1992:
            break
        if (row["date"] == num["date"]) & (row["Team A"] == num["country"]):
            a = True
            row["A_rank"] = num["ranking"]
            row["A_points"] = num["points"]
        if (row["date"] == num["date"]) & (row["TeamB"] == num["country"]):
            b = True
            row["B_rank"] = num["ranking"]
            row["B_points"] = num["points"]
        if a & b:
            # print()
            # print("Success", row["date"], " ", row["Team A"], " ", row["TeamB"])
            break
        if i == 6377:
            print(row["Team A"], " ", row["TeamB"], " ", row["date"], " ", a, " ", b)
            notFound.loc[len(notFound.index)] = [row["Team A"], row["TeamB"], row["date"], a, b]



results.to_csv('rank_added.csv',index = False)
notFound.to_csv('games_not_found.csv', index = False)
# print(results)
# print(rank)
