#!/usr/bin/env python

import os
import subprocess
import serial
import random
import MySQLdb

db = MySQLdb.connect(
	host="humidifier.cyqxc8aabmoz.us-east-2.rds.amazonaws.com",
	user="bmabetha",
	passwd="bmabetha",
	db="humidifier")

cursor = db.cursor()
cursor.execute("CREATE TABLE IF NOT EXISTS humidifier (id INT(10) NOT NULL AUTO_INCREMENT,sensor_id INT(10) NOT NULL,date TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,temp float(8,2)  DEFAULT NULL,relhum float(8,2)  DEFAULT NULL, setpoint_relhum float(8,2)  DEFAULT NULL,elapsed_time int(10) NOT NULL, by_user tinyint(1) DEFAULT 0, PRIMARY KEY (id))")
db.commit()
cursor.close()

ser = serial.Serial(
port = '/dev/tty.usbmodem1411',
baudrate = 9600
)


DEBUG = True

def logdata(sensor_id, temp, relhum, elapsed_time):
	print "sending data"
	# The first N entries of relative humidity set point are determined by the user
	cursor = db.cursor()
	cursor.execute("SELECT id, setpoint_relhum FROM humidifier ORDER BY id DESC LIMIT 1")
	if(id)
	setpoint_relhum = random.randint(40, 60)
	# for row in cursor.fetchall():
	# 	setpoint_relhum = row[0]

	cursor.execute("INSERT INTO humidifier (sensor_id, temp, relhum, elapsed_time, setpoint_relhum) VALUES (%s,%s,%s,%s,%s)", (sensor_id, temp, relhum, elapsed_time,setpoint_relhum))
	db.commit()
	cursor.close()


def run():
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
run()