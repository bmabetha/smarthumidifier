#!/usr/bin/env python

import os
import subprocess
import serial
import MySQLdb
import numpy as np
import time
import sklearn 
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.multiclass import OutputCodeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

db = MySQLdb.connect(
	host="humidifier.cyqxc8aabmoz.us-east-2.rds.amazonaws.com",
	user="bmabetha",
	passwd="bmabetha",
	db="humidifier")

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
	cursor.close()
	return tests

def getfeatures(rows):
	datapoints = []
	humoutput = []
	logging_id = []
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
		datapoints.append([month, hour, temp])
		humoutput.append(setpoint_relhum)
		logging_id.append(log_id)
	return (datapoints, humoutput, log_id)

def setpoint(rows):
	setpoint_relhumTemp = []
	setpoint_relhumTime = []
	setpoint_relhumTempTime = []

	# Create Humidity Output for differnet users
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
		if (temp == 0):
			temp = 59 
		relhumTemp = temp*0.68
		setpoint_relhumTemp.append(relhumTemp)

		# Time dependent user
		if((22 < hour <= 23) or (0 < hour <= 2)):
			relhumTime = 40

		if(10 < hour <= 14):
			relhumTime = 45

		if(14 < hour <= 18):
			relhumTime = 50

		if(18 < hour <= 22):
			relhumTime = 55

		if(6 < hour <= 10):
			relhumTime = 50

		if(2 < hour <= 6):
			relhumTime = 40
		setpoint_relhumTime.append(relhumTime)
		# Temperature and Time dependent user

		if(2 < hour <= 6):
			relhumTempTime = 30 + temp*0.15 

		if(6 < hour <= 10):
			relhumTempTime = 40 + temp*0.15 

		if(10 < hour <= 14):
			relhumTempTime = 35 + temp*0.15 

		if(14 < hour <= 18):
			relhumTempTime = 40 + temp*0.15 

		if(18 < hour <= 22):
			relhumTempTime = 45 + temp*0.15 

		if((22 < hour <= 23) or (0 < hour <= 2)):
			relhumTempTime = 30 + temp*0.15 

		setpoint_relhumTempTime.append(relhumTempTime)

	return (setpoint_relhumTemp, setpoint_relhumTime, setpoint_relhumTempTime)
		
def train():
	rows = getdata()

	(datapoints, humoutput, log_ids) = getfeatures(rows)

	X = np.asarray(datapoints)

	(setpointTemp, setpointTime, setpointTempTime) = setpoint(rows)

	Y_temp = np.asarray(setpointTemp) 
	Y_time = np.asarray(setpointTime) 
	Y_temptime = np.asarray(setpointTempTime) 

	X_traintemp, X_testtemp, Y_traintemp, Y_testtemp = train_test_split(X, Y_temp, test_size=0.33, random_state=42)
	X_traintime, X_testtime, Y_traintime, Y_testtime = train_test_split(X, Y_time, test_size=0.33, random_state=42)
	X_traintemptime, X_testtemptime, Y_traintemptime, Y_testtemptime = train_test_split(X, Y_temptime, test_size=0.33, random_state=42)

	
	# Convert floats to ints
	X_traintemp = X_traintemp.astype(int)
	X_testtemp = X_testtemp.astype(int)
	Y_traintemp = Y_traintemp.astype(int)
	Y_testtemp = Y_testtemp.astype(int)

	X_traintime = X_traintime.astype(int)
	X_testtime = X_testtime.astype(int)
	Y_traintime = Y_traintime.astype(int)
	Y_testtime = Y_testtime.astype(int)

	X_traintemptime = X_traintemptime.astype(int)
	X_testtemptime = X_testtemptime.astype(int)
	Y_traintemptime = Y_traintemptime.astype(int)
	Y_testtemptime = Y_testtemptime.astype(int)

	
	# Using SVM to train data
	

	# Training For Temp Dependent Setpoints
	clf = svm.SVC()
	clf.fit(X_traintemp, Y_traintemp)
	Y_predtemp = clf.predict(X_testtemp)
	svm_accuracytemp = accuracy_score(Y_testtemp, Y_predtemp)
	print 'SVM Temp Accurracy:',svm_accuracytemp

	
	# Training For Time Dependent Setpoints
	clf = svm.SVC()
	clf.fit(X_traintime, Y_traintime)
	Y_predtime = clf.predict(X_testtime)
	svm_accuracytime = accuracy_score(Y_testtime, Y_predtime)
	print "SVM Time Accurracy:",svm_accuracytime

	# Training For Temp and Time Dependent Setpoints
	clf = svm.SVC()
	clf.fit(X_traintemptime, Y_traintemptime)
	Y_predtemptime = clf.predict(X_testtemptime)
	svm_accuracytemptime = accuracy_score(Y_testtemptime, Y_predtemptime)
	print "SVM TempTime Accurracy:",svm_accuracytemptime
    
	# Using Random Forest to train data
	# check the best hyperparameters to use

	# Training for Temp Dependent Setpoints
	rf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=123456)
	rf.fit(X_traintemp, Y_traintemp)
	Y_predtemp = rf.predict(X_testtemp)
	rf_accuracytemp = accuracy_score(Y_testtemp, Y_predtemp)
	print "RF Temp Accurracy:",rf_accuracytemp

	# Training for Time Dependent Setpoints
	rf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=123456)
	rf.fit(X_traintime, Y_traintime)
	Y_predtime = rf.predict(X_testtime)
	rf_accuracytime = accuracy_score(Y_testtime, Y_predtime)
	print "RF Time Accurracy:",rf_accuracytime

	# Training for Temp and Time Dependent Setpoints
	rf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=123456)
	rf.fit(X_traintemptime, Y_traintemptime)
	Y_predtemptime = rf.predict(X_testtemptime)
	rf_accuracytemptime = accuracy_score(Y_testtemptime, Y_predtemptime)
	print "RF TempTime Accurracy:",rf_accuracytemptime
    

	# Using K nearest neighbors to train data 
	
	# Training for Temp Dependent Setpoints
	knn = KNeighborsClassifier(n_neighbors=6)
	knn.fit(X_traintemp, Y_traintemp)
	Y_predtemp = knn.predict(X_testtemp)
	knn_accuracytemp = accuracy_score(Y_testtemp, Y_predtemp)
	print "KNN Temp Accurracy:",knn_accuracytemp

	# Training for Time Dependent Setpoints
	knn = KNeighborsClassifier(n_neighbors=6)
	knn.fit(X_traintime, Y_traintime)
	Y_predtime = knn.predict(X_testtime)
	knn_accuracytime = accuracy_score(Y_testtime, Y_predtime)
	print "KNN Time Accurracy:",knn_accuracytime

	# Training for Temp and Time Dependent Setpoints
	knn = KNeighborsClassifier(n_neighbors=6)
	knn.fit(X_traintemptime, Y_traintemptime)
	Y_predtemptime = knn.predict(X_testtemptime)
	knn_accuracytemptime = accuracy_score(Y_testtemptime, Y_predtemptime)
	print "KNN TempTime Accurracy:",knn_accuracytemptime
	
train()	
