# Imports
from flask import Flask, render_template


# Initialize Flask Application
app = Flask(__name__)


@app.route("/") # Decorator for route / (the home page)
@app.route("/index") # Additional decorator for index route
@app.route("/index.html") # Additional decorator for index.html route
def hello():
    return render_template('index.html');

@app.route("/application")
def application():
    return "Not Implemented"


# Automatically runs debug mode if script is run directly
if __name__ == '__main__':
    app.run(debug = True)