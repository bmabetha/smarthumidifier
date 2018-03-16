import os
import subprocess
import serial
import random
import MySQLdb
import RPi.GPIO as GPIO

db = MySQLdb.connect(
	host="humidifier.cyqxc8aabmoz.us-east-2.rds.amazonaws.com",
	user="bmabetha",
	passwd="bmabetha",
	db="humidifier")

ser = serial.Serial(
port = '/dev/tty.usbmodem1411',
baudrate = 9600
)

DEBUG = True

# Set up Raspberry pi GPIO
GPIO.setup(18, GPIO.OUT)
GPIO.setmode(GPIO.BCM)

# Turn on the humidifier if the humidity is below the setpoint.

def getsetpoint():
	cursor = db.cursor()
	cursor.execute("SELECT relhum, setpoint_relhum FROM humidifier ORDER BY id DESC LIMIT 1")
	data = cursor.fetchall()
	cursor.close()
	return data

def control()
	data = getsetpoint()
	current_relhum = data[0]
	setpoint = data[1]
	if(current_relhum < setpoint):
		GPIO.output(18,GPIO.LOW)

	else
		GPIO.output(18,GPIO.HIGH)

	# Learning algorithm 
	if(readcounter == 100)