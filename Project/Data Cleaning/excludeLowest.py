# Script to exclude countries that have not played enough games

import metrics # CHECK THAT METRICS FILENAME IS RIGHT CUZ IM BAD AT CODING
import csv

lowBound = 100

cleanData = 'Clean_Data_1945Onwards.csv'
writeData = 'Clean_Data_1945andExcludeLow.csv'

with open(cleanData, newline='') as csvread:
    with open(writeData, 'w', newline='') as csvwrite:
        spamreader = csv.reader(csvread, delimiter=',')
        spamwriter = csv.writer(csvwrite, delimiter=',')

        for row in spamreader:
            if (row[0] == "date"): continue # Skip header row

            # If either country in the game has not played enough games to pass the lower bound, they are excluded from the dataset
            if (metrics.countryCountDicitonary[row[1]] < lowBound or metrics.countryCountDicitonary[row[2]] < lowBound):
                pass
            else:
                spamwriter.writerow(row)

print("Completed Exclude Lowest")