#!/usr/bin/env python

import os
import subprocess
import serial
import MySQLdb
import numpy as np
import time
import sklearn 
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.multiclass import OutputCodeClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

db = MySQLdb.connect(
	host="humidifier.cyqxc8aabmoz.us-east-2.rds.amazonaws.com",
	user="bmabetha",
	passwd="bmabetha",
	db="humidifier")

cursor = db.cursor()
cursor.execute("CREATE TABLE IF NOT EXISTS humidifier (id INT(10) NOT NULL AUTO_INCREMENT,sensor_id INT(10) NOT NULL,date TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,temp float(8,2)  DEFAULT NULL,relhum float(8,2)  DEFAULT NULL, setrelhum float(8,2)  DEFAULT NULL, elapsed_time int(10) NOT NULL,PRIMARY KEY (id))")
db.commit()
cursor.close()

DEBUG = True

def getdata():
	print "getting data"
	cursor = db.cursor()
	cursor.execute("SELECT * FROM humidifier ORDER BY id DESC")
	rows = cursor.fetchall()
	cursor.close()
	return rows

def gettest():
	cursor = db.cursor()
	cursor.execute("SELECT * FROM humidifier ORDER BY id DESC LIMIT 1")
	tests = cursor.fetchall()
	print tests
	cursor.close()
	return tests

def getfeatures(rows):
	datapoints = []
	humoutput = []
	for row in rows:
		log_id = row[0]
		sensor_id = row[1]
		date = row[2]
		year = date.year
		day = date.day
		month = date.month
		hour = date.hour
		temp = row[3]
		relhum = row[4]
		setpoint_relhum = row[5]
		elapsed_time = row[6]
		datapoints.append([month, hour, temp, relhum])
		humoutput.append(setpoint_relhum)
	return (datapoints, humoutput, log_id)
	
def learn():
	rows = getdata()
	(datapoints, humoutput, log_ids) = getfeatures(rows)

	X = np.asarray(datapoints)
	Y = np.asarray(humoutput)

	#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
	(test_datapoints, test_humoutput, log_id) = getfeatures(gettest())

	test_datapoints = np.asarray(test_datapoints)
	test_humoutput = np.asarray(test_humoutput)

	# Convert floats to ints
	X = X.astype(int)
	test_datapoints = test_datapoints.astype(int)
	Y = Y.astype(int)
	test_humoutput = test_humoutput.astype(int)

	clf = svm.SVC()
	clf.fit(X, Y)
	pred_humoutput = clf.predict(test_datapoints)
	print(pred_humoutput)
	diff = test_humoutput[0] - pred_humoutput[0]
	print(diff)	

	print log_id
	cursor = db.cursor()
	cursor.execute("UPDATE humidifier SET setpoint_relhum = %s WHERE id = %s", ([pred_humoutput[0], log_id]))
	db.commit()
	cursor.close()

while(True):	
	learn()
	# Learn every 10 minutes (ideally have the user determine the learning interval)
	t = 10
	time.sleep(t)
