"""
This script takes .wav files that are recordings of impuls series and parses each individual impuls into a seperate file.
Each recording put into the input folder is parsed into a seperate folder inside the output directory
Impulses are only parsed if the peak meets both the temporal and energetic requirements.
"""
import numpy as np
import matplotlib.pyplot as plt
import librosa as lbr
import librosa.display
import math
import shutil
import soundfile as sf
from pathlib import Path


# Returns sample numbers of all onsets that exceed the threshold value or percentage of maximum value
def FindPeaks(signal, samplingRate, thresholdPercentage=70, threshold=-1, reachBackTime=0.1, reachAheadTime=0.2):
    if threshold == -1:  # set threshold as given value or percentage of max value
        threshold = (thresholdPercentage/100) * np.amax(signal)
    detectedOnsets = librosa.onset.onset_detect(y=signal, sr=samplingRate, backtrack=False, units="samples")
    parsingIndicator = []

    for el in detectedOnsets:
        # Looking ahead and behind the automatically found onsets to include the maximum peak value
        if el < math.floor(reachBackTime * samplingRate): # If onset is too close to signal beginning
            reachBack = el
        else:
            reachBack = math.floor(reachBackTime * samplingRate)
        if len(signal) - el < math.floor(reachAheadTime * samplingRate): # If onset is too close to signal end
            reachAhead = len(signal) - el
        else:
            reachAhead = math.floor(reachAheadTime * samplingRate)

        onsetFragment = signal[list(range(el - reachBack, el + reachAhead))]

        # Discarding peaks below threshold values
        if np.amax(onsetFragment) > threshold:
            parsingIndicator.append(el)

    return parsingIndicator

# Trims peak list of all duplicates that are closer to each other then the minimalTimeDifference or closer to the end then the impulseDecayTime
def RemoveDuplicatePeaks(peaks, samplingRate, minimalTimeDifference = 1, impulseDecayTime = 0, signalLength = 0):
    for i in range(0, len(peaks)-1):  # Mark peaks that are too close to each other
        if peaks[i] + samplingRate * minimalTimeDifference > peaks[i+1]:
            peaks[i] = 0
            print('Duplicate peaks detected!')
    if signalLength != 0 and peaks[-1] + impulseDecayTime * samplingRate > signalLength:  # Mark peaks that are too close to the signal ends
        peaks[-1] = 0
        print("Peak too close to end of signal. Peak removed")
    peaks = np.ma.masked_equal(peaks,0).compressed()  # Remove marked peaks
    return peaks


# Returns array which has rows of impulses cut from the main signal based on the found peaks
def ParseImpulses(signal, samplingRate, peaks, attackTime=0.2, decayTime=7):
    impulses = []
    for el in peaks:
        if el + decayTime * samplingRate < len(signal):
            impulse = signal[list(range(el - math.floor(attackTime * samplingRate), math.floor(el + decayTime * samplingRate)))]
            if len(impulses) == 0:
                impulses = impulse
            else:
                impulses = np.vstack([impulses, impulse])
        else:
            print("Peak too close to end of signal. Impulse not parsed")
    return impulses

# Removes impulses that have too large of an energy deviation from the mean
def RemoveImpulsesWithEnergyDeviation(impulses, samplingRate, acceptableDeviation = 30, impulsePeaks = [], attackTime = 2, acceptableAttackDeviation = 60):
    impulseEnergies, attackEnergies = [], []
    for i in range (0, len(impulses[:,0])):
        impulseEnergies.append(sum((impulses[i,:]) * (impulses[i,:])))
        attackEnergies.append(sum((impulses[i,:round(attackTime*samplingRate)]) * (impulses[i,:round(attackTime*samplingRate)])))

    # Check full impulse energies
    while True:
        meanImpulseEnergy = sum(impulseEnergies) / len(impulseEnergies)
        impulsesDiscarded = 0
        for i in range(len(impulseEnergies) - 1, -1, -1):  # iterates backwards to avoid "Out of Bounds" error after values are deleted
            if impulseEnergies[i] > meanImpulseEnergy * (1 + (acceptableDeviation/100) or impulseEnergies[i] < meanImpulseEnergy * (1 - (acceptableDeviation/100))):
                # Remove impulse from all lists
                impulses = np.delete(impulses, i, axis=0)
                impulsePeaks = np.delete(impulsePeaks, i)
                impulseEnergies = np.delete(impulseEnergies, i)
                attackEnergies = np.delete(attackEnergies, i)
                impulsesDiscarded += 1
                i -= 1
                print("Energy Deviation Detected!")

        if impulsesDiscarded == 0:
            break

    # Check attack impulse energies
    while True:
        meanAttackEnergy = sum(attackEnergies) / len(attackEnergies)
        impulsesDiscarded = 0
        for i in range(len(impulseEnergies) - 1, -1, -1):  # iterates backwards to avoid "Out of Bounds" error after values are deleted
            if attackEnergies[i] < meanAttackEnergy * (1 - acceptableAttackDeviation):
                # Remove impulse from all lists
                impulses = np.delete(impulses, i, axis=0)
                impulsePeaks = np.delete(impulsePeaks, i)
                attackEnergies = np.delete(attackEnergies, i)
                impulsesDiscarded += 1
                i -= 1
                print("Early Energy Deviation Detected!")

        if impulsesDiscarded == 0:
            break

    return impulses, impulsePeaks

# Saves parsed impulses a folder inside the output directory
def WriteParsedImpulsesToFolder(impulses, sampleRate, outputDirectory, seriesDirectory):
    outputPath = Path(outputDirectory) / seriesDirectory
    Path(outputPath).mkdir()
    for i in range (0, len(impulses[:,0])):
        filename = f"{outputPath}/{seriesDirectory}_{str(i)}.wav"
        sf.write(filename, impulses[i,:], sampleRate)


# Draws the original signal with validation and discarded strips for manual validation
def DrawParsedImpulseValidation(signal, foundPeaks, validatedPeaks, samplingRate, seriesDirectory):
    # Creating found and discarded peak stripes
    parsingIndicator = np.zeros(len(signal))
    for el in foundPeaks:
        parsingIndicator[el] = 1
        parsingIndicator[el + 1] = -1

    validatedIndicator = np.zeros(len(signal))
    for el in validatedPeaks:
        validatedIndicator[el] = 1
        validatedIndicator[el + 1] = -1

    # Drawing the original signal with parsed impulse strips laid on top
    timeVector = np.arange(0, len(signal) / samplingRate, 1 / samplingRate)
    ylimit = max(abs(signal))
    plt.figure()
    plt.plot(timeVector, signal, color='silver', label='Signal')
    plt.plot(timeVector, parsingIndicator, 'orangered', label='Discarded peaks')
    plt.plot(timeVector, validatedIndicator, color='chartreuse', label='Accepted peaks')
    plt.ylim(-ylimit, ylimit)
    plt.legend(loc='lower right')
    plt.xlabel('time [s]')
    plt.title(seriesDirectory)

    plt.show()


class ParserConfig:
    def __init__(self):
        # Peak detection
        self.peakThresholdPercentage = 30  # Percent of max signal magnitude. Peaks with lower values are disregarded
        self.minimalTimeDifference = 13  # Minimal time difference between peaks for them to be included

        # Single impulse parsing
        self.attackTime = 0.2  # Time before the peak that is included in the impulse [s]
        self.decayTime = 12  # Time after the peak that is included in the impulse [s]

        # Energy validation of impulses
        self.energyDeviationPercentage = 2  # The percentage of the mean energy above which impulses are disregarded
        self.attackEnergyTime = 3  # The attack time for the purpose of impulse energy analysis
        self.attackEnergyDeviationPercentage = 2  # The percentage of the mean attack energy above which impulses are disregarded

        # Show figures of found peaks to manually check proper parsing
        self.showFoundPeaks_Flag = True


# Directories
INPUT_DIRECTORY = "ParserInputFolder"
OUTPUT_DIRECTORY = "ParserOutputFolder"


def main():
    parser_config = ParserConfig()

    if Path(OUTPUT_DIRECTORY).is_dir():
        shutil.rmtree(OUTPUT_DIRECTORY)
    Path(OUTPUT_DIRECTORY).mkdir()

    for seriesPath in sorted(Path(INPUT_DIRECTORY).iterdir()):
        if seriesPath.name != ".DS_Store":  # ignore MacOS system files
            seriesDirectory = seriesPath.stem.rstrip('.wav')

            print("Entering: " + seriesPath.name)

            signal, samplingRate = lbr.load(seriesPath, sr=None)

            foundPeaks = FindPeaks(signal, samplingRate, thresholdPercentage=parser_config.peakThresholdPercentage)
            foundPeaks = RemoveDuplicatePeaks(foundPeaks, samplingRate, parser_config.minimalTimeDifference,
                                              parser_config.decayTime, len(signal))
            foundImpulses = ParseImpulses(signal, samplingRate, foundPeaks, parser_config.attackTime, parser_config.decayTime)
            validatedImpulses, validatedPeaks = RemoveImpulsesWithEnergyDeviation(foundImpulses, samplingRate,
                                                                parser_config.energyDeviationPercentage, foundPeaks,
                                                                parser_config.attackEnergyTime, parser_config.attackEnergyDeviationPercentage)
            WriteParsedImpulsesToFolder(validatedImpulses, samplingRate, OUTPUT_DIRECTORY, seriesDirectory)

            if parser_config.showFoundPeaks_Flag:
                DrawParsedImpulseValidation(signal, foundPeaks, validatedPeaks, samplingRate, seriesDirectory)


if __name__ == "__main__":
    main()
