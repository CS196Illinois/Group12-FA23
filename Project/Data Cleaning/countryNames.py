# Purpose of this file is to create a csv dictionary for cleaning country names 
# This has been run already and the results are in Country_Names.csv
# Country_Names.csv will be used in cleaner.py to clean country names

import csv

countryNamesDictionary = {}

exceptionsDictionary = {
    "CuraÃ§ao": "Curacao",
    "RÃ©union": "Reunion Island",
    "SÃ£o TomÃ© and PrÃ­ncipe": "Sao Tome and Principe",
    "Ynys MÃ´n": "Ynys Mon",
    "Ã…land Islands": "Aland Islands",
    "FrÃ¸ya": "Froya",
    "GÄƒgÄƒuzia": "Gagauzia",
    "SzÃ©kely Land": "Szekely Land",
    "FelvidÃ©k": "Felvidek",
    "KÃ¡rpÃ¡talja": "Karpatalja",
    "SÃ¡pmi": "Sapmi",
}

with open('Results_Data_Raw.csv', newline='') as csvread:
    spamreader = csv.reader(csvread, delimiter=',')

    for row in spamreader:
        # Populate countryNamesDictionary
        if not row[1] in exceptionsDictionary.keys():
            countryNamesDictionary[row[1]] = row[1]
        else:
            countryNamesDictionary[row[1]] = exceptionsDictionary[row[1]]
        
        if not row[2] in exceptionsDictionary.keys():
            countryNamesDictionary[row[2]] = row[2]
        else:
            countryNamesDictionary[row[2]] = exceptionsDictionary[row[2]]

with open('Country_Names.csv', 'w', newline='') as csvwrite:
    spamwriter = csv.writer(csvwrite, delimiter=',')
    spamwriter.writerow(["country", "clean_name"])

    for key in countryNamesDictionary.keys():
        for letter in countryNamesDictionary[key]:
            if ord(letter) > 122:
                print(key)
        spamwriter.writerow([key, countryNamesDictionary[key]])
            