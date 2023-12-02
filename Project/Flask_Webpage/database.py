#### IMPORTS
import csv


#### GLOBAL VARIABLES
readData = 'Clean_Data_1945andExcludeLow.csv'

countryToYearToOffenseDataDictionary = {}
countryToYearToDefenseDataDictionary = {}
countryToYearToStatsDictionary = {}

startupYears = ['1945', '1946', '1947', '1948', '1949']


#### CODE BODY
# POPULATING DICITONARIES
def MakeEntry(dictionary, countryName, year, points):
    # dictionary is the country to offense/defense dictionary we are writing to
    # countryName is the name of the country
    # year is the year the game was played
    # points is the points scored or conceded
    if (countryName in dictionary):
        # If countryName already has a mapping to a dictionary, then check if the year has a mapping
        if (year in dictionary[countryName]):
            # If year mapping already exists, update the score data
            dictionary[countryName][year][0] += points
            dictionary[countryName][year][1] += 1
        else:
            # If year mapping does not exist, create the mapping
            dictionary[countryName][year] = [None] * 2
            dictionary[countryName][year][0] = points
            dictionary[countryName][year][1] = 1
    else:
        # If countryName does not have a mapping to a dictionary, then create the dictionary and year mapping
        dictionary[countryName] = {}
        dictionary[countryName][year] = [None] * 2
        dictionary[countryName][year][0] = points
        dictionary[countryName][year][1] = 1


with open(readData, newline='') as csvread:
    spamreader = csv.reader(csvread, delimiter=',')

    for row in spamreader:
        # Make Team 1 Entries
        MakeEntry(countryToYearToOffenseDataDictionary, row[1], row[0], int(row[3]))
        MakeEntry(countryToYearToDefenseDataDictionary, row[1], row[0], int(row[4]))

        # Make Team 2 Entries
        MakeEntry(countryToYearToOffenseDataDictionary, row[2], row[0], int(row[4]))
        MakeEntry(countryToYearToDefenseDataDictionary, row[2], row[0], int(row[3]))


# Accessing Data
# Helper function of stats()
def extractYear(year, home, away):
    # Setup output lists
    # Format Guide: 'A_number_games_1year', 'A_average_points_scored_1year', 'A_average_points_conceded_1year', 'A_offense_versus_global_defense_ratio_1year', 'A_defense_versus_global_offense_ratio_1year'
    homeStats = [None] * 3
    awayStats = [None] * 3

    # Extract year key from passed year integer
    yearKey = str(year)

    # Enter number of games, avg points scored, avg points conceded
    if (yearKey in countryToYearToDefenseDataDictionary[home] and yearKey in countryToYearToOffenseDataDictionary[home]):
        homeStats[0] = countryToYearToDefenseDataDictionary[home][yearKey][1]
        homeStats[1] = countryToYearToOffenseDataDictionary[home][yearKey][0] / homeStats[0]
        homeStats[2] = countryToYearToDefenseDataDictionary[home][yearKey][0] / homeStats[0]
    else:
        homeStats[0] = 0
        homeStats[1] = 0
        homeStats[2] = 0

    if (yearKey in countryToYearToDefenseDataDictionary[away] and yearKey in countryToYearToOffenseDataDictionary[away]):
        awayStats[0] = countryToYearToDefenseDataDictionary[away][yearKey][1]
        awayStats[1] = countryToYearToOffenseDataDictionary[away][yearKey][0] / awayStats[0]
        awayStats[2] = countryToYearToDefenseDataDictionary[away][yearKey][0] / awayStats[0]
    else:
        awayStats[0] = 0
        awayStats[1] = 0
        awayStats[2] = 0

    # Return the lists in a tuple
    homeStats.extend(awayStats)
    return homeStats

# Returns input for algorithm
def stats(date : int, home : str, away : str, homeField : bool):
    homeAdvantage = 0
    if (homeField):
        homeAdvantage = 1

    # Setup Keys Using date
    year1 = str(date - 1)
    year2 = str(date - 2)
    year3 = str(date - 3)
    year4 = str(date - 4)
    year5 = str(date - 5)

    # Extract Past Year Information from Dictionaries
    homeAndAwayYear1 = extractYear(year1, home, away)
    homeAndAwayYear2 = extractYear(year2, home, away)
    homeAndAwayYear3 = extractYear(year3, home, away)
    homeAndAwayYear4 = extractYear(year4, home, away)
    homeAndAwayYear5 = extractYear(year5, home, away)

    # Format Output
    output = [homeAdvantage]
    output.extend(homeAndAwayYear1)
    output.extend(homeAndAwayYear2)
    output.extend(homeAndAwayYear3)
    output.extend(homeAndAwayYear4)
    output.extend(homeAndAwayYear5)
    return output

print('Database Loaded')