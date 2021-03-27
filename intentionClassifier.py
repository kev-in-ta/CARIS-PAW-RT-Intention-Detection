"""
Author:         Kevin Ta
Date:           2021 March 13th
Purpose:        This Python script tests terrain classification by passing streamed data.
"""

# IMPORTED LIBRARIES

import os
import sys
import threading
import time
from multiprocessing import Process, Queue
from threading import Thread

import pandas as pd
from joblib import load, dump
from scipy import signal
from sklearn.metrics import accuracy_score, balanced_accuracy_score

# LOCALLY IMPORTED LIBRARIES
dir_path = os.path.dirname(os.path.realpath(__file__))

# from WheelModuleLib import *
from featuresLib import *

# DEFINITIONS

# Classification frequency
CLASS_DELAY = 0.2  # in s

# Direction Vectors
#DATA_COLUMNS = ['AngVel_L', 'AngVel_R', 'Chair_LinVel', 'Chair_AngVel', 'Torque_L', 'Torque_R', 'Torque_sum',
#                'Torque_diff', 'Torque_L_roc', 'Torque_R_roc']
DATA_COLUMNS = ['Torque_L', 'Torque_R', 'Torque_sum', 'Torque_diff', 'Torque_L_roc', 'Torque_R_roc']

EPSILON = 0.00001  # For small float values

# filter parameters
CUT_OFF = 20  # lowpass cut-off frequency (Hz)

PAD_LENGTH = 10  # pad length to let filtering be better

# DICTIONARIES

INTENTIONS_DICT = [
    ('Mahsa', 'Obstacles15', 'T3'),
    ('Mahsa', 'Obstacles35', 'T3'),
    ('Mahsa', 'RampA', 'T3'),
    ('Mahsa', 'StraightF', 'T3'),
    ('Mahsa', 'Turn90FL', 'T3'),
    ('Mahsa', 'Turn90FR', 'T3'),
    ('Mahsa', 'Turn180L', 'T3'),
    ('Mahsa', 'Turn180R', 'T3'),
    ('Jaimie', 'Obstacles15', 'T3'),
    ('Jaimie', 'Obstacles35', 'T3'),
    ('Jaimie', 'RampA', 'T3'),
    ('Jaimie', 'StraightF', 'T3'),
    ('Jaimie', 'Turn90FL', 'T3'),
    ('Jaimie', 'Turn90FR', 'T3'),
    ('Jaimie', 'Turn180L', 'T3'),
    ('Jaimie', 'Turn180R', 'T3'),
    ]

# Time domain feature functions and names
TIME_FEATURES = {'Mean': np.mean, 'Std': np.std,
                 'Max': np.amax, 'Min': np.amin, 'RMS': rms}

# TIME_FEATURES_NAMES = ['Mean', 'Std', 'Norm', 'AC', 'Max', 'Min', 'RMS', 'ZCR', 'Skew', 'EK']
TIME_FEATURES_NAMES = ['Mean', 'Std', 'Max', 'Min', 'RMS']

# Different data sets
SENSOR_MODULE = {'wLength': 16, 'fSamp': 200, 'fLow': 5, 'fHigh': 1}

PERFORMANCE = {}

#INTENTIONS_OG = {'Left': 0, 'Right': 1, 'Forward': 2, 'Stopped': 3, 'Backwards': 4}
#INTENTIONS = ['Left', 'Right', 'Forward', 'Stopped', 'Backwards']

# CLASSES

class ClIntentionDetector:
    """
    Class for establishing wireless communications.
    """

    def __init__(self, testSet, protocol='TCP'):
        """
        Purpose:	Initialize various sensors and class variables
        Passed: 	Nothing
        """

        self.testSet = testSet

        self.sensorParam = SENSOR_MODULE

        print('unpickling')

        self.RFTimelinePipeline = load('models/model.joblib')

        print(self.RFTimelinePipeline.get_params())

        self.RFResults = pd.DataFrame(columns=["Time", "Cluster 1", "Cluster 2", "Cluster 3", "Cluster 4",
                                               "Cluster 5", "Cluster 6", "Torque L", "Torque R"])

        # Prepopulate pandas dataframe
        EFTimeColumnNames = ['{} {}'.format(featName, direction) for direction in DATA_COLUMNS for
                             featName in TIME_FEATURES_NAMES]
        self.EFTimeColumnedFeatures = pd.DataFrame(data=np.zeros((1, len(EFTimeColumnNames))),
                                                   columns=EFTimeColumnNames)

        self.protocol = protocol

        # Initialize data queue and marker to pass for separate prcoesses
        self.dataQueue = Queue()
        self.runMarker = Queue()

        # Create class variables
        self.windowIMUraw = np.zeros((self.sensorParam['wLength'] + 2 * PAD_LENGTH, 6))
        self.windowIMUfiltered = np.zeros((self.sensorParam['wLength'], len(DATA_COLUMNS)))

        # Instantiate sensor information retrieval
        self.instDAQLoop = ClSensorDataStream(self.sensorParam['fSamp'], self.dataQueue, self.runMarker, self.testSet)

    def fnStart(self):
        """
        Purpose:	Intialize all active sensors in separate processed and collects data from the Queue
        Passed:		Frequency for 6-axis IMU to operate at
        """

        print('Start Process.')

        # Start terrain classification in separate thread
        intention = Thread(target=self.fnIntentionDetection, args=(CLASS_DELAY,))
        intention.start()

        timeStart = time.time()

        # Start various data collection sensors
        processDAQLoop = Process(target=self.instDAQLoop.fnRun)
        processDAQLoop.start()

        # Keep collecting data and updating rolling window
        while self.runMarker.empty():

            try:
                transmissionData = self.dataQueue.get(timeout=2)
                self.windowIMUraw = np.roll(self.windowIMUraw, -1, axis=0)
                self.windowIMUraw[-1, :] = transmissionData[:]
            except Exception as e:
                print('Exception: {}'.format(e))

        # wait for all processes and threads to complete
        intention.join()

        print("Intention detector joined.")

        print("Sensor loop joining.")
        processDAQLoop.join()
        print("Sensor loop joined.")

    def fnIntentionDetection(self, waitTime):
        """
        Purpose:	Class method for running terrain classification
        Passed:		Time in between runs
        """

        count = 0

        startTime = time.time()

        # Keep running until run marker tells to terminate
        while self.runMarker.empty():

            count += 1

            # Filter window
            self.fnFilterButter(self.windowIMUraw)

            # Build extracted feature vector
            self.fnBuildTimeFeatures(TIME_FEATURES_NAMES)

            #intentionRFTime = self.RFTimelinePipeline.predict(self.EFTimeColumnedFeatures)
            intentionRFTime = self.RFTimelinePipeline.predict_proba(self.EFTimeColumnedFeatures)

            try:
                print('Prediction: {}'.format(intentionRFTime))
                self.RFResults = self.RFResults.append({"Cluster 1": intentionRFTime[0,0],
                                                        "Cluster 2": intentionRFTime[0,1],
                                                        "Cluster 3": intentionRFTime[0,2],
                                                        "Cluster 4": intentionRFTime[0,3],
                                                        "Cluster 5": intentionRFTime[0,4],
                                                        "Cluster 6": intentionRFTime[0,5],
                                                        "Torque L": self.EFTimeColumnedFeatures['Mean Torque_L'][0],
                                                        "Torque R": self.EFTimeColumnedFeatures['Mean Torque_R'][0],
                                                        "Time": time.time()},
                                                       ignore_index=True)
            except Exception as e:
                print("Exception: {}".format(e))
                break

        # time.sleep(waitTime - (time.perf_counter() % waitTime))

        endTime = time.time()

        print("Classification Frequency: {:>8.2f} Hz. ({} Samples in {:.2f} s)".format(count / (endTime - startTime),
                                                                                       count, (endTime - startTime)))
        print("Intention Detection completed.")

        PERFORMANCE["{}-{}-{}-Classification".format(self.sensorParam['wLength'], self.testSet[1], self.testSet[0])] = (
        count, endTime - startTime)

        self.RFResults.to_csv(
            os.path.join('2021-Results',
                         "{:.0f}ms-{}-{}-{}.csv".format(CLASS_DELAY * 1000, self.sensorParam['wLength'], self.testSet[1],
                                                           self.testSet[0])))
        print('Saved.')

    def fnShutDown(self):

        print('Closing Socket')
        self.socket.close()
        try:
            self.sock.close()
        except Exception as e:
            print(e)

    def fnFilterButter(self, dataWindow):
        """
        Purpose:	Low pass butterworth filter onto rolling window and
                    stores in filtered class variable
                    Applies hanning window
        Passed:		Rolling raw IMU data
        """

        # Get normalized frequencies
        w_low = 2 * CUT_OFF / self.sensorParam['fSamp']

        # Get Butterworth filter parameters
        sos = signal.butter(N=2, Wn=w_low, btype='low', output='sos')

        dataSet = np.copy(dataWindow)

        #angVelL = signal.sosfiltfilt(sos, dataSet[:, 0])
        #angVelR = signal.sosfiltfilt(sos, dataSet[:, 1])
        #chaVelLin = signal.sosfiltfilt(sos, dataSet[:, 2])
        #chaVelAng = signal.sosfiltfilt(sos, dataSet[:, 3])
        torqueL = signal.sosfiltfilt(sos, dataSet[:, 4])
        torqueR = signal.sosfiltfilt(sos, dataSet[:, 5])

        #self.windowIMUfiltered[:, 0] = angVelL[PAD_LENGTH:self.sensorParam['wLength'] + PAD_LENGTH]
        #self.windowIMUfiltered[:, 1] = angVelR[PAD_LENGTH:self.sensorParam['wLength'] + PAD_LENGTH]
        #self.windowIMUfiltered[:, 2] = chaVelLin[PAD_LENGTH:self.sensorParam['wLength'] + PAD_LENGTH]
        #self.windowIMUfiltered[:, 3] = chaVelAng[PAD_LENGTH:self.sensorParam['wLength'] + PAD_LENGTH]
        self.windowIMUfiltered[:, 0] = torqueL[PAD_LENGTH:self.sensorParam['wLength'] + PAD_LENGTH]
        self.windowIMUfiltered[:, 1] = torqueR[PAD_LENGTH:self.sensorParam['wLength'] + PAD_LENGTH]
        self.windowIMUfiltered[:, 2] = torqueL[PAD_LENGTH:self.sensorParam['wLength'] + PAD_LENGTH] + \
                                       torqueR[PAD_LENGTH:self.sensorParam['wLength'] + PAD_LENGTH]
        self.windowIMUfiltered[:, 3] = torqueR[PAD_LENGTH:self.sensorParam['wLength'] + PAD_LENGTH] - \
                                       torqueL[PAD_LENGTH:self.sensorParam['wLength'] + PAD_LENGTH]
        self.windowIMUfiltered[:, 4] = (torqueL[PAD_LENGTH:self.sensorParam['wLength'] + PAD_LENGTH] - torqueL[
                                                                                                      PAD_LENGTH-1:
                                                                                                      self.sensorParam[
                                                                                                          'wLength'] + PAD_LENGTH - 1])
        self.windowIMUfiltered[:, 5] = (torqueR[PAD_LENGTH:self.sensorParam['wLength'] + PAD_LENGTH] - torqueR[
                                                                                                      PAD_LENGTH - 1:
                                                                                                      self.sensorParam[
                                                                                                          'wLength'] + PAD_LENGTH - 1])

    def fnBuildTimeFeatures(self, features):
        """
        Purpose:	Perform all time domain feature extraction on filtered data,
                    then columns the data
        Passed:		Feature dictionary to perform
        """
        dataList = [TIME_FEATURES[featName](self.windowIMUfiltered[:, i]) for i, direction in enumerate(DATA_COLUMNS)
                    for featName in features]
        dataNames = ['{} {}'.format(featName, direction) for direction in DATA_COLUMNS for featName
                     in features]
        self.EFTimeColumnedFeatures = pd.DataFrame(data=[dataList], columns=dataNames)


class ClSensorDataStream(threading.Timer):
    """
    Class for establishing wireless communications.
    """

    def __init__(self, frequency, dataQueue, runMarker, testSet):
        self.testSet = testSet
        self.streamFile = pd.read_excel(
            os.path.join(dir_path, "Trimmed_Data", testSet[0],
                         "{}_{}.xls".format(testSet[1], testSet[2])))
        self.streamRow = 0
        self.streamRowEnd = len(self.streamFile.index)
        self.dataQueue = dataQueue
        self.runMarker = runMarker
        self.frequency = frequency
        self.data = np.zeros(2)
        self.data_prev = np.zeros(2)

    def fnRetrieveData(self):
        """
        Purpose:	Send data to main data queue for transfer with timestamp and sensor ID.
        Passed:		None
        """

        timeRecorded = time.time()
        if self.streamRow < self.streamRowEnd:
            self.data = self.streamFile.iloc[self.streamRow, 1:7]
            #self.dataQueue.put([self.data[0], self.data[1]])
            self.dataQueue.put([self.data[0], self.data[1], self.data[2], self.data[3], self.data[4], self.data[5]])
            self.data_prev = self.data
            self.streamRow += 1
        else:
            self.runMarker.put(False)

    def fnRun(self):
        """
        Purpose:	Script that runs until termination message is sent to queue.
        Passed:		Frequency of data capture
        """

        # Sets time interval between signal capture
        waitTime = 1 / self.frequency

        # Sets trigger so code runs
        self.trigger = threading.Event()
        self.trigger.set()

        # Create repeating timer that ensures code runs at specified intervals
        timerRepeat = threading.Thread(target=self.fnRunThread, args=(waitTime,))
        timerRepeat.start()

        count = 0
        startTime = time.time()

        # Continuously reruns code and clears the trigger
        while self.runMarker.empty():
            count += 1
            self.trigger.wait()
            self.trigger.clear()
            self.fnRetrieveData()

        endTime = time.time()

        print("Sampling Frequency:       {:>8.2f} Hz. ({} Samples in {:.2f} s)".format(count / (endTime - startTime),
                                                                                       count, (endTime - startTime)))

        # Joins thread
        timerRepeat.join()

    def fnRunThread(self, waitTime):
        """
        Purpose:	Sets the trigger after waiting for specified interval
        Passed:		Interval of time to wait
        """

        while self.runMarker.empty():
            time.sleep(waitTime - (time.perf_counter() % waitTime))
            self.trigger.set()


# MAIN PROGRAM

if __name__ == "__main__":

    for testSet in INTENTIONS_DICT:

        connectedStatus = False
        processStatus = False
        runCompletion = False

        while runCompletion == False:
            try:
                instIntentionDetector = ClIntentionDetector(testSet, protocol='TCP')
                processStatus = True
                instIntentionDetector.fnStart()
                instIntentionDetector.runMarker.close()
                instIntentionDetector.dataQueue.close()
                print("Application Completed.")
                runCompletion = True
            except Exception as e:
                time.sleep(1)
                if processStatus:
                    instIntentionDetector.runMarker.put(False)
                    instIntentionDetector.fnShutDown()
                    instIntentionDetector.runMarker.close()
                    instIntentionDetector.dataQueue.close()
                    connectedStatus = False
                print(e)

        print(PERFORMANCE)

        os.makedirs(os.path.join(dir_path, '2021-Results'), exist_ok=True)
        dump(PERFORMANCE, os.path.join(dir_path, '2021-Results', 'performance.joblib'))
