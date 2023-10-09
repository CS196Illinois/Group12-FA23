# Scuffed Python code to clean/standardize data + add a column that represents score differential
# Dates are standardized to just their year
# Country names are standardized to Latin characters to increase compatibility with text display

import csv

countryNameDictionary = {}

### FOR POPULATING countryNameDictionary
with open('Country_Names.csv', newline='') as csvread:
    spamreader = csv.reader(csvread, delimiter=',')

    for row in spamreader:
        countryNameDictionary[row[0]] = row[1]

### FOR CLEANING THE DATA
with open('Results_Data_Raw.csv', newline='') as csvread:
    with open('Clean_Data_2.csv', 'w', newline='') as csvwrite:
        spamreader = csv.reader(csvread, delimiter=',')
        spamwriter = csv.writer(csvwrite, delimiter=',')

        for row in spamreader:
            # Adding Extra Column
            if row[0] == "date":
                row.append("home_minus_away")
            else:
                row.append(int(row[3]) - int(row[4]))

            # Date Cleaning
            date = None
            if '-' in row[0]:
                date = row[0].split('-')
            else:
                date = row[0].split('/')

            for number in date:
                if len(str(number)) == 4:
                    row[0] = number
            
            # Country Name Cleaning
            row[1] = countryNameDictionary[row[1]]
            row[2] = countryNameDictionary[row[2]]

            # Write to CSV File
            spamwriter.writerow(row)

print("Completed")