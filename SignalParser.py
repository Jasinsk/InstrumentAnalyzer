import numpy as np
import os
import matplotlib.pyplot as plt
import librosa as lbr
import librosa.display
import math
import shutil
import wave

# This script takes .wav files that are recordings of impuls series and parses each individual impuls into a seperate file.
# Each recording put into the input folder is parsed into a seperate folder inside the output directory
# Impulses are only parsed if the peak meets both the temporal and energetic requirements.

#returns sample numbers of all offsets that exceed threshold
def FindPeaks(signal, samplingRate, thresholdPercentage = 0.7, threshold = -1, reachBackTime = 0.1, reachAheadTime = 0.2):
    if threshold == -1:
        threshold = thresholdPercentage * np.amax(signal)
    DetectedOffsets = librosa.onset.onset_detect(signal, samplingRate, backtrack=False, units="samples")
    parsingIndicator = []

    for el in DetectedOffsets:
        if el < math.floor(reachBackTime * samplingRate):
            reachBack = el
        else:
            reachBack = math.floor(reachBackTime * samplingRate)
        if len(signal) - el < math.floor(reachAheadTime * samplingRate):
            reachAhead = len(signal) - el
        else:
            reachAhead = math.floor(reachAheadTime * samplingRate)
        offsetFragment = signal[list(range(el - reachBack, el + reachAhead))]

        if np.amax(offsetFragment) > threshold:
            parsingIndicator.append(el)

    return parsingIndicator

#trims peak list of all duplicates that are closer to each other then the minimalTimeDifference or closer to the end then the impulseDecayTime
def RemoveDuplicatePeaks(peaks, samplingRate, minimalTimeDifference = 1, impulseDecayTime = 0, signalLength = 0):
    for i in range(0, len(peaks)-1):
        if peaks[i] + samplingRate * minimalTimeDifference > peaks[i+1]:
            peaks[i] = 0
            print('Duplicate peaks detected!')
    if signalLength != 0 and peaks[-1] + impulseDecayTime * samplingRate > signalLength:
        peaks[-1] = 0
        print("Peak too close to end of signal. Peak removed")
    peaks = np.ma.masked_equal(peaks,0).compressed()
    return peaks

#returns array which has rows of impulses cut from the main signal around the found peaks
def ParseImpulses(signal, samplingRate, peaks, attackTime = 0.05, decayTime = 7):
    impulses = []
    for el in peaks:
        if el + decayTime * samplingRate < len(signal):
            impuls = signal[list(range(el - math.floor(attackTime * samplingRate), math.floor(el + decayTime * samplingRate)))]
            if impulses == []: #Not very elegant
                impulses = impuls
            else:
                impulses = np.vstack([impulses, impuls])
        else:
            print("Peak too close to end of signal. Impulse not parsed")
    return impulses

#trims impulses that have too large of a energy deviation from the mean
def RemoveImpulsesWithEnergyDeviation(impulses, samplingRate, acceptableDeviation = 0.3, impulsPeaks = [], attackTime = 2, acceptableAttackDeviation = 0.6):
    impulsEnergies, attackEnergies = [], []
    for i in range (0, len(impulses[:,0])):
        impulsEnergies.append(sum((impulses[i,:]) * (impulses[i,:])))
        attackEnergies.append(sum((impulses[i,:round(attackTime*samplingRate)]) * (impulses[i,:round(attackTime*samplingRate)])))

    ratio = 3
    while ratio !=0:
        i = 0
        meanEnergy = sum(impulsEnergies) / len(impulsEnergies)
        print("impulse mean is: " + str(meanEnergy))
        for el in impulsEnergies:
            if el > meanEnergy * (1 + ratio * acceptableDeviation) or el < meanEnergy * (1 - ratio * acceptableDeviation):
                impulses = np.delete(impulses, i, axis=0)
                impulsPeaks = np.delete(impulsPeaks, i)
                impulsEnergies = np.delete(impulsEnergies, i)
                attackEnergies = np.delete(attackEnergies, i)
                print("Energy Deviation Detected!")
            else:
                i += 1

        ratio -= 1

    ratio = 3
    while ratio !=0:
        i = 0
        meanAttackEnergy = sum(attackEnergies) / len(attackEnergies)
        print("attack mean is: " + str(meanAttackEnergy))
        for el in attackEnergies:
            if el < meanAttackEnergy * (1 - ratio * acceptableAttackDeviation):
                impulses = np.delete(impulses, i, axis=0)
                impulsPeaks = np.delete(impulsPeaks, i)
                attackEnergies = np.delete(attackEnergies, i)
                print("Early Energy Deviation Detected!")
            else:
                i += 1

        ratio -= 1

    return impulses, impulsPeaks

#pushes impulses to files in a folder inside the output folder
def WriteParsedImpulsesToFolder(impulses, sampleRate, outputDirectory, seriesDirectory):
    path = outputDirectory + "/" + seriesDirectory
    os.mkdir(path)
    for i in range (0, len(impulses[:,0])):
        filename = path + "/" + seriesDirectory + "_" + str(i) + ".wav"
        librosa.output.write_wav(filename, impulses[i,:], sampleRate)


# ----------------- Controls -------------------
# Directories
inputDirectory = "ParserInputFolder"
outputDirectory = "ParserOutputFolder"

# Peak detection
#threshold = 1
thresholdPercentage = 0.1
minimalTimeDifference = 7

# Impulse parsing
attackTime = 0.1
decayTime = 7

# Energy validation of impulses
acceptableEnergyDeviation = 0.2
attackEnergyTime = 1
attackEnergyDeviation = 0.2

# Show figures of found peaks to check correct working
showFoundPeaks_Flag = True
# ---------------------------------------------

if os.path.isdir(outputDirectory):
        shutil.rmtree(outputDirectory)
os.mkdir(outputDirectory)

for seriesSignal in os.listdir(os.fsencode(inputDirectory)):
    inputSignal = inputDirectory + "/" + os.fsdecode(seriesSignal)
    seriesDirectory = os.fsdecode(seriesSignal).rstrip('.wav')

    print("Entering: " + inputSignal)

    signal, samplingRate = lbr.load(inputSignal, sr=None)

    foundPeaks = FindPeaks(signal, samplingRate, thresholdPercentage=thresholdPercentage)
    foundPeaks = RemoveDuplicatePeaks(foundPeaks, samplingRate, minimalTimeDifference, decayTime, len(signal))
    foundImpulses = ParseImpulses(signal, samplingRate, foundPeaks, attackTime, decayTime)
    validatedImpulses, validatedPeaks = RemoveImpulsesWithEnergyDeviation(foundImpulses, samplingRate, acceptableEnergyDeviation, foundPeaks, attackEnergyTime, attackEnergyDeviation)
    WriteParsedImpulsesToFolder(validatedImpulses, samplingRate, outputDirectory, seriesDirectory)

    parsingIndicator = np.zeros(len(signal))
    for el in foundPeaks:
        parsingIndicator[el] = 1
        parsingIndicator[el+1] = -1

    validatedIndicator = np.zeros(len(signal))
    for el in validatedPeaks:
        validatedIndicator[el] = 1
        validatedIndicator[el+1]=-1

    if showFoundPeaks_Flag:
        # Drawing the results of the parsers work
        timeVector = np.arange(0, len(signal)/samplingRate, 1/samplingRate)
        ylimit = max(abs(signal)) * 1.2
        plt.figure()
        plt.plot(timeVector, signal, color='grey', label='Signal')
        plt.plot(timeVector, parsingIndicator, 'r', label='Discarded peaks')
        plt.plot(timeVector, validatedIndicator, color = 'chartreuse', label='Accepted peaks')
        plt.ylim(-ylimit, ylimit)
        plt.legend(loc='lower right')
        plt.xlabel('time [s]')
        plt.title(seriesDirectory)

        plt.show()