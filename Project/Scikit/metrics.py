# Script to spit out number of games played by each country team and in each year

import csv

filename = 'data.csv'

countryCountDicitonary = {}
yearCountDictionary = {}

### FOR POPULATING METRICS VARIABLES
with open(filename, newline='') as csvread:
    spamreader = csv.reader(csvread, delimiter=',')

    for row in spamreader:
        if (row[0] == "date"): continue # Skip header row

        # Count Year Entries
        if (row[0] in yearCountDictionary):
            yearCountDictionary[row[0]] = int(yearCountDictionary[row[0]]) + 1
        else:
            yearCountDictionary[row[0]] = 1

        # Count Country Entries
        if (row[1] in countryCountDicitonary):
            countryCountDicitonary[row[1]] = int(countryCountDicitonary[row[1]]) + 1
        else:
            countryCountDicitonary[row[1]] = 1
        
        if (row[2] in countryCountDicitonary):
            countryCountDicitonary[row[2]] = int(countryCountDicitonary[row[2]]) + 1
        else:
            countryCountDicitonary[row[2]] = 1

for key, value in countryCountDicitonary.items():
    print(key, value)
for key, value in yearCountDictionary.items():
    print(key, value)

print("Completed Metrics")