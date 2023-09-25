# Imports
from flask import Flask, render_template, url_for


# Initialize Flask Application
app = Flask(__name__)


# Flask Route Definitions
@app.route("/") # Decorator for route / (the home page)
@app.route("/index") # Additional decorator for index route
@app.route("/index.html") # Additional decorator for index.html route
def hello():
    return render_template('index.html')

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