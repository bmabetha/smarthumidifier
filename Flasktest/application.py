
import os
import subprocess
import serial
import numpy as np
import MySQLdb
import time
import sklearn 
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.multiclass import OutputCodeClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from flask import Flask, flash, redirect, render_template, request, session, url_for
from flask_session import Session
from passlib.apps import custom_app_context as pwd_context
from tempfile import mkdtemp

from helpers import *

DEBUG = True 

# configure application
app = Flask(__name__)

# ensure responses aren't cached
if app.config["DEBUG"]:
    @app.after_request
    def after_request(response):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Expires"] = 0
        response.headers["Pragma"] = "no-cache"
        return response

# configure session to use filesystem (instead of signed cookies)
app.config["SESSION_FILE_DIR"] = mkdtemp()
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# configure CS50 Library to use SQLite database
db = MySQLdb.connect(
    host="humidifier.cyqxc8aabmoz.us-east-2.rds.amazonaws.com",
    user="bmabetha",
    passwd="bmabetha",
    db="humidifier")

cursor = db.cursor()
cursor.execute("CREATE TABLE IF NOT EXISTS humidifier (id INT(10) NOT NULL AUTO_INCREMENT, sensor_id INT(10) NOT NULL,date TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,temp float(8,2)  DEFAULT NULL,relhum float(8,2)  DEFAULT NULL, setrelhum float(8,2)  DEFAULT NULL, elapsed_time int(10) NOT NULL, by_user tinyint(1) DEFAULT 0, PRIMARY KEY (id))")
db.commit()
cursor.close()

cursor = db.cursor()
cursor.execute("CREATE TABLE IF NOT EXISTS users (id INT(10) NOT NULL AUTO_INCREMENT, username VARCHAR(16) NOT NULL, hash VARCHAR(16) NOT NULL, PRIMARY KEY (id))")
db.commit()
cursor.close()


@app.route("/")
@login_required
def index():
    cursor = db.cursor()
    cursor.execute("SELECT * FROM humidifier ORDER BY id DESC LIMIT 1")
    info = cursor.fetchall()
    cursor.close()
    return render_template("setrelhum.html", info=info)


@app.route("/setrelhum", methods=["GET", "POST"])
@login_required
def setrelhum():
    """Set User Relative Humidity."""
    if request.method == "POST":
        cursor = db.cursor()
        cursor.execute("SELECT id FROM humidifier ORDER BY id DESC LIMIT 1")
        data = cursor.fetchall() 
        a = float(request.form.get("humidity"))
        setrelhum = "{0:.2f}".format(round(a,2))
        cursor.execute("UPDATE humidifier SET setpoint_relhum = %s, by_user = %s WHERE id = %s", ([setrelhum, 1, data[0][0]]))
        db.commit()
        cursor.close()
    return render_template("setrelhum.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    """Log user in."""

    # forget any user_id
    session.clear()

    # if user reached route via POST (as by submitting a form via POST)
    if request.method == "POST":

        # ensure username was submitted
        if not request.form.get("username"):
            return apology("must provide username")

        # ensure password was submitted
        elif not request.form.get("password"):
            return apology("must provide password")

        user_name = request.form.get("username")
        # query database for username
        cursor = db.cursor()
        #cursor.execute("SELECT * FROM users WHERE username = %s", (user_name))
        cursor.execute("SELECT * FROM users WHERE username LIKE (%s)", ([request.form.get("username")]))
        rows = cursor.fetchall()
        cursor.close()
        # ensure username exists and password is correct
        print rows[0][2]
        if len(rows) != 1 or not request.form.get("password", rows[0][2]):
            return apology("invalid username and/or password")

        # remember which user has logged in
        session["user_id"] = rows[0][0]

        # redirect user to home page
        return redirect(url_for("setrelhum"))

    # else if user reached route via GET (as by clicking a link or via redirect)
    else:
        return render_template("login.html")

@app.route("/logout")
def logout():
    """Log user out."""

    # forget any user_id
    session.clear()

    # redirect user to login form
    return redirect(url_for("login"))

@app.route("/register", methods=["GET", "POST"])
def register():
    """Register user."""
    session.clear()

    if request.method == "POST":
        if not request.form.get("username"):
            return apology("must provide username")

        if not request.form.get("password"):
            return apology("must provide password")

        if (request.form.get("password") != request.form.get("confirmation")):
            return apology("Passwords do not match")

        if ((request.form.get("password") == request.form.get("confirmation"))):
            cursor = db.cursor()
            cursor.execute("INSERT INTO users (username, hash) VALUES (%s,%s)", (request.form.get("username"),request.form.get("password")))
            result = cursor.fetchall()
            db.commit()
            cursor.close()
            if(result != False):
                cursor = db.cursor()
                cursor.execute("SELECT * FROM users ORDER BY id DESC LIMIT 1")
                rows = cursor.fetchall()
                id = rows[0][0]
                session["user_id"] = rows[0][0]
                cursor.close()
        return redirect(url_for("index"))
    else:
        return render_template("register.html")

if __name__ == '__main__':
    app.run(debug=True)