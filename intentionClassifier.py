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

PERSON = 'Jamie'  # 'Jamie' or 'Mahsa'

# Direction Vectors
DATA_COLUMNS = ['AngVel_L', 'AngVel_R', 'Chair_LinVel', 'Chair_AngVel', 'Torque_L', 'Torque_R']

EPSILON = 0.00001  # For small float values

# filter parameters
CUT_OFF = 20  # lowpass cut-off frequency (Hz)

PAD_LENGTH = 15  # pad length to let filtering be better

# DICTIONARIES

INTENTIONS_DICT = [
    ('Mahsa', 'Obstacles15', 'T1'),
    ('Mahsa', 'Obstacles35', 'T2'),
]

# Time domain feature functions and names
TIME_FEATURES = {'Mean': np.mean, 'Std': np.std, 'Norm': l2norm,
                 'Max': np.amax, 'Min': np.amin, 'RMS': rms}

# TIME_FEATURES_NAMES = ['Mean', 'Std', 'Norm', 'AC', 'Max', 'Min', 'RMS', 'ZCR', 'Skew', 'EK']
TIME_FEATURES_NAMES = ['Mean', 'Std', 'Norm', 'Max', 'Min', 'RMS']

# Different data sets
SENSOR_MODULE = {'wLength': 16, 'fSamp': 200, 'fLow': 20, 'fHigh': 1}

PERFORMANCE = {}

INTENTIONS_OG = {'Left': 0, 'Right': 1, 'Forward': 2, 'Stopped': 3, 'Backwards': 4}
INTENTIONS = ['Left', 'Right', 'Forward', 'Stopped', 'Backwards']

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

        self.RFResults = pd.DataFrame(columns=["True Label", "RF Time", "Time"])

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
        self.windowIMUraw = np.zeros((self.sensorParam['wLength'] + 2 * PAD_LENGTH, len(DATA_COLUMNS)))
        self.windowIMUfiltered = np.zeros((self.sensorParam['wLength'], len(DATA_COLUMNS)))

        # Instantiate sensor information retrieval
        self.instDAQLoop = ClSensorDataStream(self.sensorParam['fSamp'], self.dataQueue, self.runMarker, self.testSet)

    def fnStart(self, frequency):
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

                if transmissionData[0] in ['IMU_6', 'WHEEL']:
                    self.windowIMUraw = np.roll(self.windowIMUraw, -1, axis=0)
                elif transmissionData[0] in ['USS_DOWN', 'USS_FORW']:
                    pass
                elif transmissionData[0] in ['PI_CAM']:
                    pass
            except Exception as e:
                print(e)

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

            intentionRFTime = self.RFTimelinePipeline.predict(self.EFTimeColumnedFeatures)

            try:
                print('Prediction: {0:>10s}'.format(INTENTIONS[intentionRFTime[0]]))
                self.RFResults = self.RFResults.append({"Cluster Label": INTENTIONS_OG[self.testSet[1]],
                                                        "RF Time": intentionRFTime[0], "Time": time.time()},
                                                       ignore_index=True)
            except Exception as e:
                print(e)
                break

        # time.sleep(waitTime - (time.perf_counter() % waitTime))

        endTime = time.time()

        print("Classification Frequency: {:>8.2f} Hz. ({} Samples in {:.2f} s)".format(count / (endTime - startTime),
                                                                                       count, (endTime - startTime)))
        print("Intention Detection completed.")

        PERFORMANCE["{}-{}-{}-Classification".format(self.testSet[1], self.testSet[2], self.testSet[0])] = (
        count, endTime - startTime)

        self.RFResults.to_csv(
            os.path.join('2021-Results',
                         "{:.0f}ms-{}-{}-{}.csv".format(CLASS_DELAY * 1000, self.testSet[1], self.testSet[2],
                                                           self.testSet[0])))
        print('Saved.')

        self.RFResults = self.RFResults[self.RFResults["RF Time"] != 0]

        y_pred = self.RFResults["RF Time"].to_numpy(dtype=np.int8)
        print(y_pred.shape)
        y_test = INTENTIONS_OG[self.testSet[1]] * np.ones(len(y_pred), dtype=np.int8)
        print(y_test.shape)

        print(accuracy_score(y_test, y_pred))
        print(balanced_accuracy_score(y_test, y_pred))

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

        # Filter all the data columns
        for i in range(6):
            self.windowIMUfiltered[:, i] = signal.sosfiltfilt(sos, dataSet[:, i])[
                                           PAD_LENGTH:self.sensorParam['wLength'] + PAD_LENGTH]  # *hanningWindow

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
        self.streamFile = pd.read_csv(
            os.path.join(dir_path, "Trimmed_Data", PERSON,
                         "Middle_{}Power{}{}_Module.csv".format(testSet[1], testSet[2], testSet[0])))
        self.streamRow = 0
        self.streamRowEnd = len(self.streamFile.index)
        self.dataQueue = dataQueue
        self.runMarker = runMarker
        self.offset = np.zeros(6)
        self.frequency = frequency

    def fnRetrieveData(self):
        """
        Purpose:	Send data to main data queue for transfer with timestamp and sensor ID.
        Passed:		None
        """

        timeRecorded = time.time()
        if self.streamRow < self.streamRowEnd:
            data = self.streamFile.iloc[self.streamRow, :]
            self.dataQueue.put(['IMU_6', data[9], data[0], data[1], data[2], data[3], data[4], data[5]])
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

        PERFORMANCE["{}-{}-{}-Acquisition".format(self.testSet[1], self.testSet[2], self.testSet[0])] = (
            count, endTime - startTime)

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

        dump(PERFORMANCE, os.path.join('2021-Results', 'performance.joblib'))
