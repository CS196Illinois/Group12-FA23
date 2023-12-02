# Imports
from flask import Flask, render_template, url_for
import database
import model


# Initialize Flask Application
app = Flask(__name__)

# Recent Calculations
TIE_IMAGE = "static/NoLoss.png"
HOME_WIN_IMAGE = "static/RightLoss.png"
AWAY_WIN_IMAGE = "static/LeftLoss.png"
JERSEY_ONE = "static/flags/England.png"
JERSEY_TWO = "static/flags/New_Zealand.png"

calculations = [
    {
        'year': 'YYYY',
        'home': 'HOME',
        'away': 'AWAY',
        'field' :'HOME FIELD',
        'percentages': [50, 50],
        'home_jersey': JERSEY_ONE,
        'away_jersey': JERSEY_TWO,
        'background': TIE_IMAGE
    },
    {
        'year': 'YYYY',
        'home': 'HOME',
        'away': 'AWAY',
        'field' : 'NEUTRAL FIELD',
        'percentages': [70, 30],
        'home_jersey': JERSEY_ONE,
        'away_jersey': JERSEY_TWO,
        'background': HOME_WIN_IMAGE
    },
    {
        'year': 'YYYY',
        'home': 'HOME',
        'away': 'AWAY',
        'field' : 'NEUTRAL FIELD',
        'percentages': [40, 50],
        'home_jersey': JERSEY_ONE,
        'away_jersey': JERSEY_TWO,
        'background': AWAY_WIN_IMAGE
    }
]


# Flask Route Definitions
@app.route("/") # Decorator for route / (the home page)
@app.route("/index") # Additional decorator for index route
@app.route("/index.html") # Additional decorator for index.html route
def hello():
    return render_template('index.html', calcs=calculations)

@app.route("/howitworks")
@app.route("/howitworks.html")
@app.route("/how_it_works")
def howitworks():
    return render_template('howitworks.html')

@app.route("/application")
def application():
    return render_template('application.html')

@app.route("/application/<home>/<away>/<year>/<field>")
def submit(home, away, year, field):
    percentages = model.LogReg.predict_proba([database.stats(int(year), home, away, field == "HOME")])[0]
    percentages[0] = round(percentages[0] * 100, 1)
    percentages[1] = round(percentages[1] * 100, 1)

    background = "/static/NoLoss.png"
    if (percentages[1] - percentages[0] >= 5):
        background = "/static/RightLoss.png"
    elif (percentages[0] - percentages[1] >= 5):
        background = "/static/LeftLoss.png"

    home_jersey = "/static/flags/" + "_".join(home.split(" ")) + ".png"
    away_jersey = "/static/flags/" + "_".join(away.split(" ")) + ".png"

    return render_template('application.html', params={
        'home' : home,
        'away' : away,
        'year' : year,
        'field' : field + " FIELD",
        'percentages' : percentages,
        'background' : background,
        'home_jersey' : home_jersey,
        'away_jersey' : away_jersey
    })


# Automatically runs debug mode if script is run directly
if __name__ == '__main__':
    app.run(debug = True)