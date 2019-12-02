import numpy as np
import os
import matplotlib.pyplot as plt
import librosa as lbr
import librosa.display
import math

def PeakFinder(signal, samplingRate, threshold = -1, reachBackTime = 0.05, reachAheadTime = 0.1): #returns sample numbers of all offsets that exceed threshold
    if threshold == -1:
        threshold = 0.6 * np.amax(signal)
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

def RemoveDuplicatePeaks(peaks, samplingRate, minimalTimeDifference = 5):
    for i in range(0, len(peaks) - 1):
        if peaks[i] + samplingRate * minimalTimeDifference > peaks[i+1]:
            peaks[i+1] = 0
    peaks = np.ma.masked_equal(peaks,0).compressed()
    return peaks

def ParseImpulses(signal, samplingRate, peaks, attackTime = 0.05, decayTime = 7):
    impulses = []
    for el in peaks:
        impuls = signal[list(range(el - math.floor(attackTime * samplingRate), math.floor(el + decayTime * samplingRate)))]
        if impulses == []: #Not very elegant
            impulses = impuls
        else:
            impulses = np.vstack([impulses, impuls])
    return impulses

def RemoveImpulsesWithEnergyDeviation(impulses, acceptableDeviation = 0.3):
    impulsEnergies = []
    for i in range (0, len(impulses[:,0])):
        impulsEnergies.append(sum((impulses[i,:]) * (impulses[i,:])))
    meanEnergy = sum(impulsEnergies)/len(impulsEnergies)
    i = 0
    for el in impulsEnergies:
        if el > meanEnergy * (1 + acceptableDeviation) or el < meanEnergy * (1 - acceptableDeviation):
            impulses = np.delete(impulses, i, axis=0)
            print("Energy Deviation!")
        else:
            i += 1
    return impulses

def WriteParsedImpulsesToFolder(impulses, sampleRate, folderName):
    path = folderName
    os.mkdir(path)
    for i in range (0, len(impulses[:,0])):
        filename = folderName + "/" + outputFolderName + "_" + str(i) + ".wav"
        librosa.output.write_wav(filename, impulses[i,:], sampleRate)

# main
inputDirectoryName = "SignalFolder"
inputDirectory = os.fsencode(inputDirectoryName)

for seriesSignal in os.listdir(inputDirectory):
    inputSignal = inputDirectoryName + "/" + os.fsdecode(seriesSignal)
    outputFolderName = os.fsdecode(seriesSignal).rstrip('.wav')

    signal, samplingRate = lbr.load(inputSignal)

    foundPeaks = PeakFinder(signal, samplingRate)
    foundPeaks = RemoveDuplicatePeaks(foundPeaks, samplingRate)
    foundImpulses = ParseImpulses(signal, samplingRate, foundPeaks)
    validatedImpulses = RemoveImpulsesWithEnergyDeviation(foundImpulses, 0.17)
    WriteParsedImpulsesToFolder(validatedImpulses, samplingRate, outputFolderName)

#print(foundPeaks)

    parsingIndicator = np.zeros(len(signal))
    for el in foundPeaks:
        parsingIndicator[el] = 1

    plt.figure()
    librosa.display.waveplot(signal, samplingRate)
    librosa.display.waveplot(parsingIndicator, samplingRate)
    plt.title(outputFolderName)
    plt.show()
'''
    plt.figure()
    librosa.display.waveplot(validatedImpulses[0,:], samplingRate)
    plt.title('Signal')
    plt.show()
'''