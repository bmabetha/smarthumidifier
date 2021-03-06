#!/usr/bin/env python

import os
import subprocess
import serial
import MySQLdb
import numpy as np
import time
import random
import matplotlib.pyplot as plt
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
	passwd="***",
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
		relhumTemp = temp*0.68 + random.uniform(0, 1)
		setpoint_relhumTemp.append(relhumTemp)

		# Time dependent user
		if((22 < hour <= 23) or (0 < hour <= 2)):
			relhumTime = 40 + random.uniform(0, 1)

		if(10 < hour <= 14):
			relhumTime = 45 + random.uniform(0, 1)

		if(14 < hour <= 18):
			relhumTime = 50 + random.uniform(0, 1)

		if(18 < hour <= 22):
			relhumTime = 55 + random.uniform(0, 1)

		if(6 < hour <= 10):
			relhumTime = 50 + random.uniform(0, 1)

		if(2 < hour <= 6):
			relhumTime = 40 + random.uniform(0, 1)
		setpoint_relhumTime.append(relhumTime)
		# Temperature and Time dependent user

		if(2 < hour <= 6):
			relhumTempTime = 30 + temp*0.15 

		if(6 < hour <= 10):
			relhumTempTime = 40 + temp*0.15 + random.uniform(0, 1)

		if(10 < hour <= 14):
			relhumTempTime = 35 + temp*0.15 + random.uniform(0, 1)

		if(14 < hour <= 18):
			relhumTempTime = 40 + temp*0.15 + random.uniform(0, 1)

		if(18 < hour <= 22):
			relhumTempTime = 45 + temp*0.15 + random.uniform(0, 1)

		if((22 < hour <= 23) or (0 < hour <= 2)):
			relhumTempTime = 30 + temp*0.15 + random.uniform(0, 1)

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

	# Check for overfitting
	Y_predtemptrain = clf.predict(X_traintemp)
	svm_accuracytemptrain = accuracy_score(Y_traintemp, Y_predtemptrain)
	print 'SVM Temp Train Accurracy:',svm_accuracytemptrain

	
	# Training For Time Dependent Setpoints
	clf = svm.SVC()
	clf.fit(X_traintime, Y_traintime)
	Y_predtime = clf.predict(X_testtime)
	svm_accuracytime = accuracy_score(Y_testtime, Y_predtime)
	print "SVM Time Accurracy:",svm_accuracytime

	# Check for overfitting
	Y_predtimetrain = clf.predict(X_traintime)
	svm_accuracytimetrain = accuracy_score(Y_traintime, Y_predtimetrain)
	print 'SVM Time Train Accurracy:',svm_accuracytimetrain

	# Training For Temp and Time Dependent Setpoints
	clf = svm.SVC()
	clf.fit(X_traintemptime, Y_traintemptime)
	Y_predtemptime = clf.predict(X_testtemptime)
	svm_accuracytemptime = accuracy_score(Y_testtemptime, Y_predtemptime)
	print "SVM TempTime Accurracy:",svm_accuracytemptime

	# Check for overfitting
	Y_predtemptimetrain = clf.predict(X_traintemptime)
	svm_accuracytemptimetrain = accuracy_score(Y_traintemptime, Y_predtemptimetrain)
	print 'SVM TempTime Train Accurracy:',svm_accuracytemptimetrain
    
	# Using Random Forest to train data
	# check the best hyperparameters to use
	

	# Training for Temp Dependent Setpoints
	rf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=123456)
	rf.fit(X_traintemp, Y_traintemp)
	Y_predtemp = rf.predict(X_testtemp)
	rf_accuracytemp = accuracy_score(Y_testtemp, Y_predtemp)
	print "RF Temp Accurracy:",rf_accuracytemp

	# Check for overfitting
	Y_predtemptrain = rf.predict(X_traintemp)
	rf_accuracytemptrain = accuracy_score(Y_traintemp, Y_predtemptrain)
	print 'RF Temp Train Accurracy:',rf_accuracytemptrain

	# Training for Time Dependent Setpoints
	rf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=123456)
	rf.fit(X_traintime, Y_traintime)
	Y_predtime = rf.predict(X_testtime)
	rf_accuracytime = accuracy_score(Y_testtime, Y_predtime)
	print "RF Time Accurracy:",rf_accuracytime

	# Check for overfitting
	Y_predtimetrain = rf.predict(X_traintime)
	rf_accuracytimetrain = accuracy_score(Y_traintime, Y_predtimetrain)
	print 'RF Time Train Accurracy:',rf_accuracytimetrain

	# Training for Temp and Time Dependent Setpoints
	rf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=123456)
	rf.fit(X_traintemptime, Y_traintemptime)
	Y_predtemptime = rf.predict(X_testtemptime)
	rf_accuracytemptime = accuracy_score(Y_testtemptime, Y_predtemptime)
	print "RF TempTime Accurracy:",rf_accuracytemptime

	# Check for overfitting
	Y_predtemptimetrain = rf.predict(X_traintemptime)
	rf_accuracytemptimetrain = accuracy_score(Y_traintemptime, Y_predtemptimetrain)
	print 'RF TempTime Train Accurracy:',rf_accuracytemptimetrain
    

	# Using K nearest neighbors to train data 
	
	# Training for Temp Dependent Setpoints
	knn = KNeighborsClassifier(n_neighbors=6)
	knn.fit(X_traintemp, Y_traintemp)
	Y_predtemp = knn.predict(X_testtemp)
	knn_accuracytemp = accuracy_score(Y_testtemp, Y_predtemp)
	print "KNN Temp Accurracy:",knn_accuracytemp

	# Check for overfitting
	Y_predtemptrain = knn.predict(X_traintemp)
	knn_accuracytemptrain = accuracy_score(Y_traintemp, Y_predtemptrain)
	print 'KNN Temp Train Accurracy:',knn_accuracytemptrain

	# Training for Time Dependent Setpoints
	knn = KNeighborsClassifier(n_neighbors=6)
	knn.fit(X_traintime, Y_traintime)
	Y_predtime = knn.predict(X_testtime)
	knn_accuracytime = accuracy_score(Y_testtime, Y_predtime)
	print "KNN Time Accurracy:",knn_accuracytime

	# Check for overfitting
	Y_predtimetrain = knn.predict(X_traintime)
	knn_accuracytimetrain = accuracy_score(Y_traintime, Y_predtimetrain)
	print 'KNN Time Train Accurracy:',knn_accuracytimetrain

	# Training for Temp and Time Dependent Setpoints
	knn = KNeighborsClassifier(n_neighbors=6)
	knn.fit(X_traintemptime, Y_traintemptime)
	Y_predtemptime = knn.predict(X_testtemptime)
	knn_accuracytemptime = accuracy_score(Y_testtemptime, Y_predtemptime)
	print "KNN TempTime Accurracy:",knn_accuracytemptime

	# Check for overfitting
	Y_predtemptimetrain = knn.predict(X_traintemptime)
	knn_accuracytemptimetrain = accuracy_score(Y_traintemptime, Y_predtemptimetrain)
	print 'KNN TempTime Train Accurracy:',knn_accuracytemptimetrain

	accuracy = [svm_accuracytemp, rf_accuracytemp, knn_accuracytemp, svm_accuracytime, rf_accuracytime, \
				knn_accuracytime, svm_accuracytemptime, rf_accuracytemptime, knn_accuracytemptime]
	accuracy = [x * 100 for x in accuracy]
	
	
	x = np.arange(len(accuracy))
	barlist = plt.bar(x, accuracy)
	plt.xticks(x+.5, ['SVM Temp','RF Temp','KNN Temp', 'SVM Time','RF Time', 'KNN Time', 'SVM TempTime','RF TempTime','KNN TempTime'])
	barlist[0].set_color('r')
	barlist[1].set_color('b')
	barlist[2].set_color('g')
	barlist[3].set_color('r')
	barlist[4].set_color('b')
	barlist[5].set_color('g')
	barlist[6].set_color('r')
	barlist[7].set_color('b')
	barlist[8].set_color('g')
	plt.title('Accuracy performance of learning algorithms')
	plt.ylabel('Accuracy (%)')
	plt.grid()
	plt.show()

	overfitting = [svm_accuracytemp, svm_accuracytemptrain, rf_accuracytemp, rf_accuracytemptrain, knn_accuracytemp, knn_accuracytemptrain]
	overfitting = [x * 100 for x in overfitting]

	otherprofiles = [svm_accuracytime, svm_accuracytimetrain, rf_accuracytime, rf_accuracytimetrain, knn_accuracytime, knn_accuracytimetrain, \
					svm_accuracytemptime, svm_accuracytemptimetrain, rf_accuracytemptime, rf_accuracytemptimetrain, knn_accuracytemptime, knn_accuracytemptimetrain]
	
	
	x = np.arange(len(overfitting))
	barlist = plt.bar(x, overfitting)
	plt.xticks(x+.5, ['SVM Temp','SVM TempTrain','RF Temp', 'RF TempTrain','KNN Temp', 'KNN TempTrain'])
	barlist[0].set_color('r')
	barlist[1].set_color('r')
	barlist[2].set_color('b')
	barlist[3].set_color('b')
	barlist[4].set_color('g')
	barlist[5].set_color('g')
	plt.title('Overfitting Analysis for Temp Dependent Prediction')
	plt.ylabel('Accuracy (%)')
	plt.grid()
	plt.show()
	
train()	
