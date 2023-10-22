# Script to extract data from the 5 columns of human readable data

import csv

# Filenames
readData = 'Clean_Data_1945andExcludeLow.csv'
# writeData = 'Extracted_Data.csv'
writeData = 'Extracted_Data_WithMissing.csv'

# Dictionaries
# yearToGlobalOffenseDictionary = {}
# yearToGlobalDefenseDictionary = {}
countryToYearToOffenseDataDictionary = {}
countryToYearToDefenseDataDictionary = {}
# Oh my god these two last ones are so scuffed
# Ok, let me try and explain these two countryToYearToOffenseDataDictionary things.
# They should really be their own classes but I'm lazy and this code will get used by literally only me so we don't exactly need readability for modification
# The structure is as follows:
# country name will be mapped to a dictionary 
# that dictionary in turn maps years to an array that holds their data in [total points earned/conceded, #games]
# This is objectively horrible btw and should not be replicated. Do as I say, not as I do.

# We need information from 5 years in the past so the first 5 years are used as startup data but omitted
startupYears = ['1945', '1946', '1947', '1948', '1949']


### POPULATING DICTIONARIES ###
# Helper function for populating so I don't develop a crippling mental illness when trying to debug
def MakeEntry(dictionary, countryName, year, points):
    # dictionary is the scuffed country dictionary we are writing to
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

    currentYear = None
    totalYearlyGames = 0
    totalYearlyPointsScored = 0
    totalYearlyPointsConceded = 0

    for row in spamreader:
        if (currentYear == None): currentYear = row[0] # Setup the first iterated year
        # if (row[0] != currentYear): 
            # If a year changes, we parse and save the yearly data
            # yearToGlobalOffenseDictionary[currentYear] = totalYearlyPointsScored / totalYearlyGames
            # yearToGlobalDefenseDictionary[currentYear] = totalYearlyPointsConceded / totalYearlyGames

            # Update currentYear and reset the yearly data
            # currentYear = row[0] 
            # totalYearlyGames = 0
            # totalYearlyPointsScored = 0
            # totalYearlyPointsConceded = 0

        # Increment totalYearlyGames
        # totalYearlyGames += 1
        
        # Make Team 1 Entries
        MakeEntry(countryToYearToOffenseDataDictionary, row[1], row[0], int(row[3]))
        MakeEntry(countryToYearToDefenseDataDictionary, row[1], row[0], int(row[4]))

        # Make Team 2 Entries
        MakeEntry(countryToYearToOffenseDataDictionary, row[2], row[0], int(row[4]))
        MakeEntry(countryToYearToDefenseDataDictionary, row[2], row[0], int(row[3]))
    
    # Handle Final Year's Data
    # yearToGlobalOffenseDictionary[currentYear] = totalYearlyPointsScored / totalYearlyGames
    # yearToGlobalDefenseDictionary[currentYear] = totalYearlyPointsConceded / totalYearlyGames

### WRITING DATA FILE ###
# Helper of the helper function lol
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
    # else:
        # homeStats[0] = 0
        # homeStats[1] = 0
        # homeStats[2] = 0

    if (yearKey in countryToYearToDefenseDataDictionary[away] and yearKey in countryToYearToOffenseDataDictionary[away]):
        awayStats[0] = countryToYearToDefenseDataDictionary[away][yearKey][1]
        awayStats[1] = countryToYearToOffenseDataDictionary[away][yearKey][0] / awayStats[0]
        awayStats[2] = countryToYearToDefenseDataDictionary[away][yearKey][0] / awayStats[0]
    # else:
        # awayStats[0] = 0
        # awayStats[1] = 0
        # awayStats[2] = 0

    # Enter offense and defense ratios and rescale to percentage difference divided by 10
    # globalOffense = yearToGlobalOffenseDictionary[yearKey]
    # globalDefense = yearToGlobalDefenseDictionary[yearKey]

    # homeStats[3] = (homeStats[1] / globalDefense - 1) * 10
    # homeStats[4] = (homeStats[2] / globalOffense - 1) * 10
    # awayStats[3] = (awayStats[1] / globalDefense - 1) * 10
    # awayStats[4] = (awayStats[2] / globalOffense - 1) * 10

    # Return the lists in a tuple
    return (homeStats, awayStats)

# Another helper function to preserve sanity
def formatRow(row):
    # Extract Basic Information From row
    date = int(row[0])
    home = row[1]
    away = row[2]

    homeAdvantage = 0
    awayAdvantage = 0
    if (row[5] == 'FALSE'):
        homeAdvantage = 1
        awayAdvantage = -1
    
    homeMinusAway = int(row[3]) - int(row[4])
    awayMinusHome = homeMinusAway * -1

    homeResult = 'Tie'
    awayResult = 'Tie'
    if (homeMinusAway > 0):
        homeResult = 'Win'
        awayResult = 'Lose'
    elif (homeMinusAway < 0):
        homeResult = 'Lose'
        awayResult = 'Win'

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

    # Format output lists
    homeAsTeamARow = [date, home, away, row[3], row[4], homeAdvantage, homeMinusAway, homeResult] + homeAndAwayYear1[0] + homeAndAwayYear1[1] + homeAndAwayYear2[0] + homeAndAwayYear2[1] + homeAndAwayYear3[0] + homeAndAwayYear3[1] + homeAndAwayYear4[0] + homeAndAwayYear4[1] + homeAndAwayYear5[0] + homeAndAwayYear5[1]
    awayAsTeamARow = [date, away, home, row[4], row[3], awayAdvantage, awayMinusHome, awayResult] + homeAndAwayYear1[1] + homeAndAwayYear1[0] + homeAndAwayYear2[1] + homeAndAwayYear2[0] + homeAndAwayYear3[1] + homeAndAwayYear3[0] + homeAndAwayYear4[1] + homeAndAwayYear4[0] + homeAndAwayYear5[1] + homeAndAwayYear5[0]

    # Return lists in a tuple
    return (homeAsTeamARow, awayAsTeamARow)

with open(readData, newline='') as csvread:
    with open(writeData, 'w', newline='') as csvwrite:
        spamreader = csv.reader(csvread, delimiter=',')
        spamwriter = csv.writer(csvwrite, delimiter=',')

        header = [
            'date', 'Team A', 'TeamB', 'A_score', 'B_score', 'A_field_advantage', 'A_minus_B', 'A_result',
            'A_number_games_1year', 'A_average_points_scored_1year', 'A_average_points_conceded_1year', # 'A_offense_versus_global_defense_ratio_1year', 'A_defense_versus_global_offense_ratio_1year',
            'B_number_games_1year', 'B_average_points_scored_1year', 'B_average_points_conceded_1year', # 'B_offense_versus_global_defense_ratio_1year', 'B_defense_versus_global_offense_ratio_1year',
            'A_number_games_2year', 'A_average_points_scored_2year', 'A_average_points_conceded_2year', # 'A_offense_versus_global_defense_ratio_2year', 'A_defense_versus_global_offense_ratio_2year',
            'B_number_games_2year', 'B_average_points_scored_2year', 'B_average_points_conceded_2year', # 'B_offense_versus_global_defense_ratio_2year', 'B_defense_versus_global_offense_ratio_2year',
            'A_number_games_3year', 'A_average_points_scored_3year', 'A_average_points_conceded_3year', # 'A_offense_versus_global_defense_ratio_3year', 'A_defense_versus_global_offense_ratio_3year',
            'B_number_games_3year', 'B_average_points_scored_3year', 'B_average_points_conceded_3year', # 'B_offense_versus_global_defense_ratio_3year', 'B_defense_versus_global_offense_ratio_3year',
            'A_number_games_4year', 'A_average_points_scored_4year', 'A_average_points_conceded_4year', # 'A_offense_versus_global_defense_ratio_4year', 'A_defense_versus_global_offense_ratio_4year',
            'B_number_games_4year', 'B_average_points_scored_4year', 'B_average_points_conceded_4year', # 'B_offense_versus_global_defense_ratio_4year', 'B_defense_versus_global_offense_ratio_4year',
            'A_number_games_5year', 'A_average_points_scored_5year', 'A_average_points_conceded_5year', # 'A_offense_versus_global_defense_ratio_5year', 'A_defense_versus_global_offense_ratio_5year',
            'B_number_games_5year', 'B_average_points_scored_5year', 'B_average_points_conceded_5year', # 'B_offense_versus_global_defense_ratio_5year', 'B_defense_versus_global_offense_ratio_5year'
        ]
        spamwriter.writerow(header)

        for row in spamreader:
            if (row[0] == 'date'): continue # Skip Header
            if (row[0] in startupYears): continue # Skip Startup Years
            pair = formatRow(row)

            spamwriter.writerow(pair[0])
            spamwriter.writerow(pair[1])


print("Completed Extracting Data")