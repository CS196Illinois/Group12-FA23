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
JERSEY_ONE = "static/jerseyone.png"
JERSEY_TWO = "static/jerseytwo.png"

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


# Automatically runs debug mode if script is run directly
if __name__ == '__main__':
    app.run(debug = True)