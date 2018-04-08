# smarthumidifier
Building a learning humidifier

This project builds a humidifier system that adjusts its output to match the user's relative humidity preference. 
It is composed of Transmitter Stations:
  Humidity/temperature sensor, 
  RF transmitter, 
  Microcontroller - Arduino
This is for measuring features that will be used in predicting the relative humidity.

Receiver Station
  RF receiver,
  Microcontroller - Raspberry Pi

Data collected from the sensors is transferred to the receiver station, uploaded to Amazon database. A learning algorigthm runs
a server to determine the relative humidity output based on the current temperature, time, day, and month. The algorithm used in
this case is Random Forest. After training different models on data from UCI (https://archive.ics.uci.edu/ml/datasets/SML2010), 
Random Forest performed better than KNN and SVM. 
The predicted relative humidity is updated in the server and used to control the solid state realy switch connected
to the humidifier until the room reaches the desired/predicted realtive humidity.
