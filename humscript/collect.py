#!/usr/bin/env python

import os
import subprocess
import serial
import random
import MySQLdb

#from controlpi import * 

db = MySQLdb.connect(
	host="humidifier.cyqxc8aabmoz.us-east-2.rds.amazonaws.com",
	user="bmabetha",
	passwd="***",
	db="humidifier")

cursor = db.cursor()
cursor.execute("CREATE TABLE IF NOT EXISTS humidifier (id INT(10) NOT NULL AUTO_INCREMENT,sensor_id INT(10) NOT NULL,date TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,temp float(8,2)  DEFAULT NULL,relhum float(8,2)  DEFAULT NULL, setpoint_relhum float(8,2)  DEFAULT NULL,elapsed_time int(10) NOT NULL, by_user tinyint(1) DEFAULT 0, PRIMARY KEY (id))")
db.commit()
cursor.close()

ser = serial.Serial(
#port = '/dev/ttyACM0', # When using raspberry pi	
port = '/dev/tty.usbmodem1411', # When using arduino to laptop
baudrate = 9600
)

DEBUG = True

def logdata(sensor_id, temp, relhum, elapsed_time):
	print "sending data"

	# The first N entries of relative humidity set point are determined by the user
	# Remember to put condition of setting setpoint for the first N entries

	# Time dependent user

	# Select the last setpoint and use that to determine the set point of the new entry
	cursor = db.cursor()
	cursor.execute("SELECT id, setpoint_relhum FROM humidifier ORDER BY id DESC LIMIT 1")
	data = cursor.fetchall()
	setpoint_relhum = 40;

	cursor.execute("INSERT INTO humidifier (sensor_id, temp, relhum, elapsed_time, setpoint_relhum) VALUES (%s,%s,%s,%s,%s)", (sensor_id, temp, relhum, elapsed_time,setpoint_relhum))
	db.commit()
	cursor.close()


def run():
	prev_rel = 0  # initialize previous relative humidity to be zero
	firstreading = 0 # Track First Entry to have it as the start of the time stamp
	while(True):
		line = ser.readline().strip()
		if len(line) > 0:
			if DEBUG:
				print "Received input: "+line
			splitList = line.split(":")
			values = dict(zip(splitList[0::2],splitList[1::2]))
			if values.has_key('ID') and values.has_key('TS'):
				if firstreading == 0:
					# keep track of the first time
					firstreading = 1
					start = values['TS']
				if values['ID']:
					sensor = values['ID']
					end = values['TS']
					elapsed_time = int(end)  - int(start)
					logdata(sensor_id=values['ID'], temp=values['TF'], relhum=values['RH'], elapsed_time=elapsed_time)
		'''
		if(prev != values['RH']):
			prev = values['RH']
			control()  # since control is an expensive call, only do it if there is a change in relative humidity		
		'''
run()
