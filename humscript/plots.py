#!/usr/bin/env python

import os
import subprocess
import serial
import random
import MySQLdb
import numpy as np
import matplotlib.pyplot as plt
import random

db = MySQLdb.connect(
	host="humidifier.cyqxc8aabmoz.us-east-2.rds.amazonaws.com",
	user="bmabetha",
	passwd="***",
	db="humidifier")

DEBUG = True

def getdata():
	print "getting data"
	cursor = db.cursor()
	cursor.execute("SELECT * FROM humidifier WHERE id BETWEEN 1763 AND 2101")
	rows = cursor.fetchall()
	cursor.close()
	return rows

def getfeatures(rows):
	datapoints = []
	setpoint_hum = []
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
		setpoint_hum.append(setpoint_relhum)

	return (datapoints, setpoint_hum, log_id)

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
		relhumTemp = temp*0.68 + random.uniform(0, 5)
		setpoint_relhumTemp.append(relhumTemp)

		# Time dependent user
		if((22 < hour <= 23) or (0 < hour <= 2)):
			relhumTime = 40 + random.uniform(0, 5)

		if(10 < hour <= 14):
			relhumTime = 45 + random.uniform(0, 5)

		if(14 < hour <= 18):
			relhumTime = 50 + random.uniform(0, 5)

		if(18 < hour <= 22):
			relhumTime = 55 + random.uniform(0, 5)

		if(6 < hour <= 10):
			relhumTime = 50 + random.uniform(0, 5)

		if(2 < hour <= 6):
			relhumTime = 40 + random.uniform(0, 5)
		setpoint_relhumTime.append(relhumTime)
		# Temperature and Time dependent user

		if(2 < hour <= 6):
			relhumTempTime = 30 + temp*0.15 

		if(6 < hour <= 10):
			relhumTempTime = 40 + temp*0.15 + random.uniform(0, 5)

		if(10 < hour <= 14):
			relhumTempTime = 35 + temp*0.15 + random.uniform(0, 5)

		if(14 < hour <= 18):
			relhumTempTime = 40 + temp*0.15 + random.uniform(0, 5)

		if(18 < hour <= 22):
			relhumTempTime = 45 + temp*0.15 + random.uniform(0, 5)

		if((22 < hour <= 23) or (0 < hour <= 2)):
			relhumTempTime = 30 + temp*0.15 + random.uniform(0, 5)

		setpoint_relhumTempTime.append(relhumTempTime)

	return (setpoint_relhumTemp, setpoint_relhumTime, setpoint_relhumTempTime)

def controlplot():
	rows = getdata()
	(datapoints, setpoint_hum, log_ids) = getfeatures(rows)
	features = np.asarray(datapoints)
	#setpoint_hum = np.asarray(setpoint_hum)

	x = np.linspace(0,15*len(features[:,1]),len(features[:,1]))
	room_hum = features[:,3].transpose()
	hum_setpoint = np.ones((len(features[:,1]),), dtype=np.int)
	hum_setpoint = 50*hum_setpoint

	
	plt.plot(x, room_hum)
	plt.plot(x, hum_setpoint)
	plt.title('Humidity With Humidifier On')
	plt.ylabel('Relative Humidity (%)')
	plt.xlabel('Time (s)')
	plt.grid()
	plt.show()
def webdataplot():
	'''
	with open("textFile.txt") as textFile:
    	lines = [line.split() for line in textFile]
	'''

	lines = map(str.split, open('NEW-DATA-1.T15.txt'))

	'''
	text_file = open("NEW-DATA-1.T15.txt", "r")
	#lines = text_file.read().split("\n")
	lines = text_file.read().split("\n")
	#features = lines.strip(" ")
	'''
	print len(lines)

	print type(lines[1][0])
	print lines[0]
	print lines[1]
	print lines[1][3]

webdataplot()


