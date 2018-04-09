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
import pandas as pd 
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.multiclass import OutputCodeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# This training analysis uses training data from online. https://archive.ics.uci.edu/ml/datasets/SML2010
def setpoint(datapoints):
	setpoint_relhumTemp = []
	setpoint_relhumTime = []
	setpoint_relhumTempTime = []

	# Create Humidity Output for differnet users
	for datapoint in datapoints:
		hour = datapoint[0]
		day = datapoint[1]
		temp = datapoint[2]
		hum = datapoint[3]
		light = datapoint[4] 
		#relative humidity for a Temp dependent user
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



def onlinedata():
	lines1 = map(str.split, open('NEW-DATA-1.T15.txt'))
	lines2 = map(str.split, open('NEW-DATA-2.T15.txt'))
	features1 = np.asarray(lines1)
	features2 = np.asarray(lines2)
	#print features2[1:1372,:].shape
	#features = np.concatenate((features1, features2[1:1372,:]), axis=1)
	print len(features1)
	print features2.shape
	datapoints1 = []
	for feature in features1:
		time = feature[1].split(":")
		hour = time[0]
		temp = feature[3]
		hum = feature[8]
		light = feature[10]
		day = feature[23]
		datapoints1.append([hour,day,temp,hum,light])

	datapoints2 = []
	for feature in features2:
		time = feature[1].split(":")
		hour = time[0]
		temp = feature[3]
		hum = feature[8]
		light = feature[10]
		day = feature[23]
		datapoints2.append([hour,day,temp,hum,light])

	datapoints1 = np.asarray(datapoints1)
	datapoints2 = np.asarray(datapoints2)

	datapoints = np.concatenate((datapoints1[1:2765,:], datapoints2[1:1374,:]), axis=0)
	
	print datapoints[0]
	print datapoints.shape
	datapoints = [map(float, x) for x in datapoints]
	datapoints = np.asarray(datapoints)
	print datapoints.shape

	
	(setpointTemp, setpointTime, setpointTempTime) = setpoint(datapoints)

	X = datapoints

	print "Shape of X:", X.shape

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
	plt.title('Accuracy of the Learning Algorithms for Different User Profiles')
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
	plt.title('Overfitting Analysis for Temp Dependent Prediction for a Time Dependent User')
	plt.ylabel('Accuracy (%)')
	plt.grid()
	plt.show()
	print type(datapoints[1][2])
	print datapoints[1]

	svm_average_accuracy = (svm_accuracytemp + svm_accuracytime + svm_accuracytemptime)/3
	rf_average_accuracy = (rf_accuracytemp + rf_accuracytime + rf_accuracytemptime)/3
	knn_average_accuracy = (knn_accuracytemp + knn_accuracytime + knn_accuracytemptime)/3

	print 'SVM Average Accuracy:',svm_average_accuracy
	print 'RF Average Accuracy:',rf_average_accuracy
	print 'KNN Average Accuracy:',knn_average_accuracy

	svm_tempoverfitting = svm_accuracytemptrain - svm_accuracytemp
	rf_tempoverfitting = rf_accuracytemptrain - rf_accuracytemp
	knn_tempoverfitting = knn_accuracytemptrain - knn_accuracytemp

	svm_timeoverfitting = svm_accuracytimetrain - svm_accuracytime
	rf_timeoverfitting = rf_accuracytimetrain - rf_accuracytime
	knn_timeoverfitting = knn_accuracytimetrain - knn_accuracytime

	svm_temptimeoverfitting = svm_accuracytemptimetrain - svm_accuracytemptime
	rf_temptimeoverfitting = rf_accuracytemptimetrain - rf_accuracytemptime
	knn_temptimeoverfitting = knn_accuracytemptimetrain - knn_accuracytemptime

	svm_overfitting = (svm_tempoverfitting + svm_timeoverfitting + svm_temptimeoverfitting)/3
	rf_overfitting = (rf_tempoverfitting + rf_timeoverfitting + rf_temptimeoverfitting)/3
	knn_overfitting = (knn_tempoverfitting + knn_timeoverfitting + knn_temptimeoverfitting)/3

	print 'SVM Temp Overfitting:',svm_tempoverfitting
	print 'RF Temp Overfitting:',rf_tempoverfitting
	print 'KNN Temp Overfitting:',knn_tempoverfitting

	print 'SVM Time Overfitting:',svm_timeoverfitting
	print 'RF Time Overfitting:',rf_timeoverfitting
	print 'KNN Time Overfitting:',knn_timeoverfitting

	print 'SVM Time Temp Overfitting:',svm_temptimeoverfitting
	print 'RF Time Temp Overfitting:',rf_temptimeoverfitting
	print 'KNN Time Temp Overfitting:',knn_temptimeoverfitting

	print 'SVM Overfitting:',svm_overfitting
	print 'RF Overfitting:',rf_overfitting
	print 'KNN Overfitting:',knn_overfitting

	averageaccuracy = [svm_average_accuracy, rf_average_accuracy, knn_average_accuracy]
	averageaccuracy = [x * 100 for x in averageaccuracy]

	x = np.arange(len(averageaccuracy))
	barlist = plt.bar(x, averageaccuracy)
	plt.xticks(x+.5, ['SVM Ave','RF Ave','KNN Ave'])
	barlist[0].set_color('r')
	barlist[1].set_color('b')
	barlist[2].set_color('g')
	plt.title('Average Accuracy Across User Profiles')
	plt.ylabel('Accuracy (%)')
	plt.grid()
	plt.show()

	averageoverfitting = [svm_overfitting, rf_overfitting, knn_overfitting]
	averageoverfitting = [x * 100 for x in averageoverfitting]

	x = np.arange(len(averageoverfitting))
	barlist = plt.bar(x, averageoverfitting)
	plt.xticks(x+.5, ['SVM OverfittingAve','RF OverfittingAve','KNN OverfittingAve'])
	barlist[0].set_color('r')
	barlist[1].set_color('b')
	barlist[2].set_color('g')
	plt.title('Average Overfitting Across User Profiles')
	plt.ylabel('Accuracy (%)')
	plt.grid()
	plt.show()
	


onlinedata()