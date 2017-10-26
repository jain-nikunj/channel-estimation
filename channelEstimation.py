import numpy as np
import math
from random import random
import matplotlib.pyplot as plt
from numpy import sin, linspace, pi, cos
from scipy import fft, arange
import sys

def init_params():
    '''All global parameters initialized here'''
    global noiseThreshold, maxLagEnd, minTaps, maxTaps
    global maxErrorToleranceThreshold, lagSearchWindow, numSamplesUsed
    global dummyOffset, maxToleranceThreshold, corrOffset, fs, correlationDict
    global toneFrequencies, numTonePeriods, lenTone, runMode, maxSamples
    global maxChannelLen, numSamples, fileSourceBitsExpt1
    global fileSinkComplexExpt2,fileSinkComplexExpt1, fileSourceComplexExpt2
    global xStartSequence, xEndSequence


    runMode = 2 #0: before expt1, 1: before expt2, 2: after expt2
    maxSamples = 2000000 # Length of bit sequence of 0,1 BPSK used for expt 1
    numSamples = 2000 #Length of bit sequence used in input for expt 2
    numSamplesUsed = 1500 #M for channel estimate
    maxChannelLen = 400 #Max number of samples in channel response
    maxLagEnd = 10000
    fs = 500e3; #Sampling frequency in Hz

    minTaps = 1
    maxTaps = 10
    corrOffset = 0
    maxToleranceThreshold = 0.995 #Set this depending on noise
    maxErrorToleranceThreshold = 0.99
    dummyOffset = 500 #Dummy values in output to simulate practical scenario while receiving
    toneFrequencies = [ fs/100.0, fs/200.0, fs/500.0]
    numTonePeriods = 5 #Number of periods for lowest frequency 
    lenTone = int(numTonePeriods*fs//min(toneFrequencies));
    lagSearchWindow = 20; #Search around estimated lag in window of this size

    noiseThreshold = 1e-1
    xStartSequence = 0
    xEndSequence = int((maxSamples - numSamplesUsed)*1e-2)

    correlationDict = {} #key: lag for this sine wave #value: the sine wave
    #For files
    fileDirectory = "./exptfiles"
    fileSourceBitsExpt1 = "fileSourceBitsExpt1" #Binary file used as (byte) file source for expt 1
    fileSinkComplexExpt1 = "fileSinkComplexExpt1" #Binary file containing (complex) file sink dump from expt 1
    fileSinkComplexExpt2 = "fileSinkComplexExpt2" #Binary file containing (complex) file sink dump from expt 2

    fileSourceComplexExpt2 = "fileSourceComplexExpt2" #Binary file used as (complex) file source for expt 2


    fileSourceBitsExpt1  = fileDirectory + "/" + fileSourceBitsExpt1
    fileSinkComplexExpt1  = fileDirectory + "/" + fileSinkComplexExpt1
    fileSinkComplexExpt2  = fileDirectory + "/" + fileSinkComplexExpt2

    fileSourceComplexExpt2 = fileDirectory + "/" + fileSourceComplexExpt2

#Before expt 1
def get_bits_data():
    '''Returns bit array to be written to file to be used as file source for gnuradio expt1 '''
    bitsData = [int(255*random()) for sample in range(maxSamples)]

    return bitsData

def write_bits_input():
    '''Writes the byte array to be used as (byte) file source for gnuradio expt 1 '''

    bitsData = get_bits_data()

    #Write to file here
    with open(fileSourceBitsExpt1, 'wb') as newFile:
      newFileByteArray = bytearray(bitsData)
      newFile.write(newFileByteArray)
      newFile.close()


#Before expt 2
def get_complex_data():
    """Returns complex values corresponding to input read from file sink of
    expt 1 """
    # floatsIn = np.fromfile(fileSourceComplexExpt2)
    # complexData = np.array([complex(floatsIn[2*i], floatsIn[2*i + 1]) for i in range(len(floatsIn) // 2)])

    complexData = np.fromfile(fileSinkComplexExpt1, dtype = 'complex64')
    complexData = complexData[1:numSamplesUsed]
    # plot.plot(complexData.real)
    # plt.title('expt1 output')
    # plt.show()


    return complexData

def get_complex_tones():
    '''Returns complex values corresponding to input sine tones '''
    # complexTones = []
    # N = 1600 #length of channel
    # f = 1.0/N
    # x = np.array(range(2*N))*(2*pi*f)
    # y = sin(x)  + 0

    y = []
    x = np.array(range(lenTone))
    for k,toneFrequency in enumerate(toneFrequencies):
        curStart = k*lenTone + corrOffset
        yTemp = np.array(sin(x*2.0*pi*toneFrequency/fs))
        correlationDict[toneFrequency] = {}
        correlationDict[toneFrequency]['start'] = curStart
        correlationDict[toneFrequency]['wave'] = yTemp[corrOffset:]
        for i in range(len(yTemp)):
            y.append(yTemp[i])

    complexTones = np.array(y, dtype = 'complex64')

    # complexTones = np.zeros((1,2*len(y)))
    # for i in range(len(y)):
    #     complexTones[0][2*i] = y[i]
    #     complexTones[0][2*i+1] = 0

    # logFile.write(complexTones)

    # for start in correlationDict:
    #     logFile.write(start)
    #     plt.plot(correlationDict[start])
    #     plt.title(str(start))
    #     plt.show()
    return complexTones


def write_complex_input():
    '''Writes the byte array for tone + data into file to be used as (complex) file source for gnuradio for expt2 '''

    complexTones = get_complex_tones()
    complexData = get_complex_data()


    complexInput = np.concatenate([complexTones,complexData] , axis = 1)
    plt.plot(complexInput.real)
    plt.show()
    #Write to file here  
    with open(fileSourceComplexExpt2, 'wb') as newFile:
        # complexData.tofile(fileSourceComplexExpt2)
        # logFile.write(complexTones)
        complexInput.tofile(fileSourceComplexExpt2)



#After expt 2
def correlate(x, y):
    '''Returns x.y/sqrt(x.x*y.y) '''
    if(len(x) == len(y)):
        num = np.sum(np.multiply(x,y))
        den = np.sqrt(np.sum(np.multiply(x,x))*np.sum(np.multiply(y,y)))
        if den is not 0:
            return num/den
        else:
            return 1
    else:
        raise ValueError("Length of x = " + str(len(x)) + \
          " does not match length of  y = "  + str(len(y)))

def read_output():
    '''Reads the complex values from output file sink generated by gnuradio expt 2'''

    complexOutput = np.fromfile(
      fileSinkComplexExpt2, dtype = 'complex64').reshape(-1,1)
    # logFile.write(complexOutput.shape)

    complexOutput = np.concatenate(
      [np.array(np.zeros((dummyOffset,1))), complexOutput], axis = 0)
    # logFile.write(complexOutput.shape)
    # plt.plot(complexOutput.real)
    # plt.title('expt2')
    # plt.show()
    return complexOutput


def get_max_range(x, tolerance):
    '''Returns the indices and values for all values >= tolerance*maximum'''
    maxIndices = []
    maxRange = []
    maxVal = max(x)

    for k,elem in enumerate(x):
        if elem >= tolerance*maxVal:
            maxRange.append(elem)
            maxIndices.append(k)

    return [maxIndices, maxRange]

def get_min_range(x, tolerance):
    '''Returns the indices and values for all values <= maximum/tolerance'''
    minIndices = []
    minRange = []
    minVal = min(x)

    for k,elem in enumerate(x):
        if elem <= minVal/tolerance:
            minRange.append(elem)
            minIndices.append(k)

    return [minIndices, minRange]

def get_lag_estimate(complexOutput):
    '''Returns best lag estimate based on coarse timing synchronization using tones'''

    lagStart = dummyOffset//2;
    lagEstArray = []
    for toneFrequency in toneFrequencies:
        start = correlationDict[toneFrequency]['start']
        curX = correlationDict[toneFrequency]['wave']
        logFile.write ('Correlating for wave starting at ' + str(start) + '...'+ "\n")

        lagEnd = min(maxLagEnd, len(complexOutput) - len(curX) )


        corrArray = []
        lagArray = []
        for lag in arange(lagStart, lagEnd+1):
            curY = complexOutput[lag:lag+len(curX), 0].real

            curCorr = correlate(abs(curX), abs(curY))
            corrArray.append(curCorr)
            lagArray.append(lag)
            # logFile.write ('Lag: ' + str(lag) + ', Corr: ' + str(curCorr))

        [maxIndices, maxRange] = get_max_range(corrArray, maxToleranceThreshold)

        curLagArray = []
        for index in maxIndices:
            curLagArray.append(lagArray[index])

        logFile.write('---Max Lags'+ "\n")
        logFile.write('---' + str(curLagArray)+ "\n")
        logFile.write('---Max Values'+ "\n")
        logFile.write('---'+ str(maxRange)+ "\n")


        curEstLag = np.median(curLagArray) - start
        logFile.write('---Current Estimated Lag: ' + str(curEstLag)+ "\n")
        lagEstArray.append(curEstLag)
        # plt.plot(arange(lagStart,lagEnd), corrArray)
        # plt.show()
    lagEstimate = int(np.mean(lagEstArray))
    logFile.write('Final estimated lag: ' + str(lagEstimate)+ "\n")
    return lagEstimate


def get_channel_estimate_with_numtaps(
    numTaps, lagEstimate, complexInput, complexOutput, xOffset):
    #Scan a range around the lag
    channelArray = []
    errorArray = []
    lagArray = []
    for lag in arange(lagEstimate - lagSearchWindow//2, 1+lagEstimate + lagSearchWindow//2):
        # logFile.write('---Considering lag ' + str(lag) + '...')
        xStart = lenTone*len(toneFrequencies) + maxChannelLen//2 + xOffset
        xEnd = xStart + numSamplesUsed + numTaps -1

        yStart = xStart + lag + numTaps - 1
        yEnd = xEnd + lag
        if xEnd < len(complexInput):
            curX = complexInput[xStart:xEnd]
        else:
            raise ValueError('Index out of bounds in complex Input,' + \
              'StartIndex: ' + str(xStart) + ', EndIndex: ' + str(xEnd) + \
              ' Length: ' + str(len(complexInput)))

        if yStart >= 0 and yEnd < len(complexOutput):
            curY = complexOutput[yStart:yEnd]
        else:
            raise ValueError('Index out of bounds in complex Output' + \
              ', StartIndex: ' + str(yStart) + ', EndIndex: ' + str(yEnd) + \
              ' Length: ' + str(len(complexOutput)))


        # plt.plot(range(len(curX)), curX.real, 'b', range(len(curY)), curY.real, 'r')
        # plt.show()

        matrix = []
        for i in range(numSamplesUsed):
          matrix.append([])
          for j in range(numTaps):
            matrix[i].append(curX[i + numTaps - 1 - j])

        matrix = np.array(matrix)
        matrixpinv = np.linalg.pinv(matrix)


        curChannelEstimate = np.dot(matrixpinv, curY)
        # logFile.write('------Channel Estimate: ')
        # logFile.write(curChannelEstimate)

        curError = np.linalg.norm(np.dot(matrix,curChannelEstimate) - curY,2)**2
        # logFile.write('------Error:' + str(curError))

        lagArray.append(lag)
        errorArray.append(curError)
        channelArray.append(curChannelEstimate)

    [minIndices, minRange] = get_min_range(errorArray, maxErrorToleranceThreshold)

    # logFile.write(minRange)
    # logFile.write(minIndices)
    yError = min(minRange)
    lenFlatRegion = len(minRange)
    channel = channelArray[minIndices[0]]

    return [yError, lenFlatRegion, channel]

def get_channel_estimate():
    '''Returns estimated channel based on matrix inversion and searching a range around lag estimates'''

    complexTones = get_complex_tones()
    complexData = get_complex_data()
    complexInput = np.concatenate([complexTones,complexData] , axis = 0)
    complexOutput = read_output()

    lagEstimate = get_lag_estimate(complexOutput)

    for i in range(xStartSequence, xEndSequence - numSamplesUsed):
        channelEstimate = []
        tapArray = []
        errorArray = []
        lenFlatRegionArray = []
        channelArray = []
        for numTaps in arange(minTaps,maxTaps+1):
            logFile.write ('Getting channel estimate assuming ' + str(numTaps) + ' taps...'+ "\n")

            [yError, lenFlatRegion, channel] = get_channel_estimate_with_numtaps(
              numTaps, lagEstimate, complexInput, complexOutput, i)
            logFile.write ('Error: ' + str(yError)+ "\n")
            logFile.write('Length of flat region: ' + str(lenFlatRegion)+ "\n")

            tapArray.append(numTaps)
            errorArray.append(yError)
            lenFlatRegionArray.append(lenFlatRegion)
            channelArray.append(channel)

        # plt.plot(tapArray, errorArray, 'r')
        # plt.title('Error vs numTaps')
        # plt.ylim((0, 1e1*noiseThreshold))
        # plt.savefig('Er')
        # plt.show()
        # plt.plot(tapArray, lenFlatRegionArray, 'b')
        # plt.title('Length of flat region vs numTaps')
        # plt.show()

        #Get minimum index so that length of flat region is 2
        minIndex = -1
        predictedTaps = -1

        errorChangeArray = [0]
        for i in range(1,len(tapArray)):
           errorChange = abs(errorArray[i]-errorArray[i-1])/errorArray[i-1]
           errorChangeArray.append(errorChange)
           logFile.write(tapArray[i], i, errorArray[i], errorArray[i-1], errorChange)
           if errorChange <= 0.1 and abs(errorArray[i]) < noiseThreshold:
            minIndex = i - 1
            predictedTaps = tapArray[i-1]
            break
        # plt.plot(tapArray, errorChangeArray, 'r')
        # plt.title('Error change vs numTaps')
        # plt.show()
        channelEstimate = channelArray[minIndex]
        logFile.write("Timestep ({})".format(str(i)) + "\n")
        logFile.write("Error V/S Number of Taps:" + "\n")
        logFile.write(str(errorArray)+ "\n")
        logFile.write("Min Index: " + str(minIndex)+ "\n")
        logFile.write("Predicted numTaps: " + str(predictedTaps)+ "\n")
        logFile.write("Channel Estimate:"+ "\n")
        logFile.write(str(channelEstimate)+ "\n")

def run():
    '''Call functions depending on modes'''
    if runMode is 0: #Before expt 1
        logFile.write("Writing byte array bit input..."+ "\n")
        write_bits_input()
    elif runMode is 1: #Before expt 2
        logFile.write("Writing byte array complex input..."+ "\n")
        write_complex_input()
    elif runMode is 2: #After expt 2
        logFile.write("Getting channel estimate..."+ "\n")
        get_channel_estimate()
    else:
        raise ValueError('Unsupported runMode ' + str(runMode)+ "\n")

def main():
    global logFile
    with open("channelEstimation.log", "w") as logFile: #OVERWRITES
        init_params()
        run()
        logFile.write("Done!"+ "\n")

if __name__ == "__main__":
    main()
