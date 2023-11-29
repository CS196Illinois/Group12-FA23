# To extract relevant data from the fifa rank dataset
import csv
import pandas as pd

rank = pd.read_csv("fifa_ranking.csv", parse_dates=['rank_date'])

# reformat date
rank['rank_date'] = rank['rank_date'].dt.year
rank.pop('country_abrv')
# rank.pop('cur_year_avg')
# rank.pop('cur_year_avg_weighted')
# rank.pop('two_year_ago_avg')
# rank.pop('two_year_ago_weighted')
# rank.pop('three_year_ago_avg')
# rank.pop('three_year_ago_weighted')
rank.pop('confederation')
rank.pop('previous_points')
rank.pop('rank_change')
# rank.pop('last_year_avg')
# rank.pop('last_year_avg_weighted')
# print(rank)

# sort according to country and year, take the average
# print(rank)
rank = rank.sort_values(by=['country_full', 'rank_date'])

# with open('averaged.csv', 'w', newline='') as csvfile:
#     filewriter = csv.writer(csvfile, delimiter=',',
#                             quotechar='|', quoting=csv.QUOTE_MINIMAL)
#     filewriter.writerow(['rank', 'country', 'total_points','rank_year'])

dict = {'ranking': [],
        'country': [],
        'points': [],
        'date': []
        }
averaged = pd.DataFrame(dict)

rank = pd.DataFrame(rank)

c = "Afghanistan"
d = 2003
num = 0
totp = 0.0
totr = 0
for index, row in rank.iterrows():
    if (row["rank_date"] == d) & (row["country_full"] == c):
        num += 1
        totp += row["total_points"]
        totr += (row["rank"])
    else:
        if c == 'Congo DR':
            averaged.loc[len(averaged.index)] = [round(int(totr / num), 0), 'DR Congo', round(totp / num, 1), d]
        elif c == 'Côte d\'Ivoire':
            averaged.loc[len(averaged.index)] = [round(int(totr / num), 0), 'Ivory Coast', round(totp / num, 1), d]
        elif c == 'USA':
            averaged.loc[len(averaged.index)] = [round(int(totr / num), 0), 'United States', round(totp / num, 1), d]
        elif c == 'St. Vincent / Grenadines':
            averaged.loc[len(averaged.index)] = [round(int(totr / num), 0), 'Saint Vincent and the Grenadines', round(totp / num, 1), d]
        elif c == 'St. Vincent and the Grenadines':
            averaged.loc[len(averaged.index)] = [round(int(totr / num), 0), 'Saint Vincent and the Grenadines', round(totp / num, 1), d]
        elif c == 'St Lucia':
            averaged.loc[len(averaged.index)] = [round(int(totr / num), 0), 'Saint Lucia', round(totp / num, 1), d]
        elif c == 'St. Lucia':
            averaged.loc[len(averaged.index)] = [round(int(totr / num), 0), 'Saint Lucia', round(totp / num, 1), d]
        elif c == 'IR Iran':
            averaged.loc[len(averaged.index)] = [round(int(totr / num), 0), 'Iran', round(totp / num, 1), d]
        elif c == 'Chinese Taipei':
            averaged.loc[len(averaged.index)] = [round(int(totr / num), 0), 'Taiwan', round(totp / num, 1), d]
        elif c == 'Cape Verde Island':
            averaged.loc[len(averaged.index)] = [round(int(totr / num), 0), 'Cape Verde', round(totp / num, 1), d]
        elif c == 'Cabo Verde':
            averaged.loc[len(averaged.index)] = [round(int(totr / num), 0), 'Cape Verde', round(totp / num, 1), d]
        elif c == 'Korea DPR':
            averaged.loc[len(averaged.index)] = [round(int(totr / num), 0), 'North Korea', round(totp / num, 1), d]
        elif c == 'Korea Republic':
            averaged.loc[len(averaged.index)] = [round(int(totr / num), 0), 'South Korea', round(totp / num, 1), d]
        elif c == 'St. Kitts and Nevis':
            averaged.loc[len(averaged.index)] = [round(int(totr / num), 0), 'Saint Kitts and Nevis', round(totp / num, 1), d]
        else:
            averaged.loc[len(averaged.index)] = [round(int(totr/ num), 0), c, round(totp/ num, 1), d]
        c = row["country_full"]
        d = row["rank_date"]
        num = 1
        totp = 0.0
        totr = 0.0


rank = rank.sort_values(by=['country_full', 'rank_date'])

rank.to_csv('rank_simplified.csv',index = False)
averaged.to_csv('averaged.csv',index = True)
# print(averaged)

