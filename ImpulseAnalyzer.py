import numpy as np
from scipy import signal
import os
import matplotlib.pyplot as plt
import librosa
import librosa.display
import shutil
import csv

# This script accepts folders with individually parsed impulses and calculates an averaged spectrum for each directory.
# The plots of these results are saved into the output directory.
# Spectral and energy parameters are also calculated and saved into the output directory.

def InsertIntoVstack(vector, stack):
    if stack == []:  # Not very elegant way to make sure the first impulse is loaded in correctly
        stack = vector
    else:
        stack = np.vstack([stack, vector])

    return stack

def CalculateAverageVector(Vectors): #Takes a vector of vectors and calculates a average vector
    averageVector = np.mean(Vectors, axis=0)
    return averageVector

def CalculateFFTs(takes, samplingRate, attackTime, sustainTime): #Takes array of impulses and creates array of spectrums
    attackSpectrums, sustainSpectrums, decaySpectrums = [], [], []
    attackFrequencies, sustainFrequencies, decayFrequencies = 0, 0, 0
    for take in takes:
        attackFrequencies, attackSpectrum = signal.periodogram(take[:int(attackTime*samplingRate)], samplingRate, scaling="spectrum")
        sustainFrequencies, sustainSpectrum = signal.periodogram(take[int(attackTime*samplingRate):int(sustainTime*samplingRate)], samplingRate, scaling="spectrum")
        decayFrequencies, decaySpectrum = signal.periodogram(take[int(sustainTime*samplingRate):], samplingRate, scaling="spectrum")

        attackSpectrums = InsertIntoVstack(attackSpectrum.real, attackSpectrums)
        sustainSpectrums = InsertIntoVstack(sustainSpectrum.real, sustainSpectrums)
        decaySpectrums = InsertIntoVstack(decaySpectrum.real, decaySpectrums)

    return attackFrequencies, attackSpectrums, sustainFrequencies, sustainSpectrums, decayFrequencies, decaySpectrums

def CalculateRMS(signal):
    return np.sum(librosa.feature.rmse(signal))

def CalculateDecayTime(impulse, samplingRate, windowLength = 1000, ratio = 10): #WIP: this is extremally bodged together
    envelope = []
    decayTime = 0
    maxEnv = 0
    peakTime = 0
    for i in range(0,len(impulse)):

        energy = CalculateRMS(impulse[i:i+windowLength])
        if energy > maxEnv:
            maxEnv = energy
            peakTime = i

        if energy * ratio < maxEnv:
            decayTime = (i - peakTime)/samplingRate
            #print('Decay Time: ' + str(decayTime))
            break

    return decayTime

def run(inputDirectory, outputDirectory, parameterFileName, spectrumFileName, fileNameAppendix, attackTime, sustainTime, decayTime_flag):

    if os.path.isdir(outputDirectory):
            shutil.rmtree(outputDirectory)
    os.mkdir(outputDirectory)

    samplingRate = 0
    # These variables are used to save the parameter data to a csv file
    data_array = []
    series_names, centroid_values, centroid_deviations, rolloff_values, rolloff_deviations, rms_values, rms_deviations, \
        bandwidth_values, bandwidth_deviation, zeroCrossingRate_values, zeroCrossingRate_deviations, decayTime_values, decayTime_deviations, tuning_values, tuning_deviations = [" "], ["Spectrum Centroid"], ["Centroid Deviation"], \
                                                        ["Rolloff"], ["Rolloff Deviation"], ["RMS"], ["RMS Deviation"], \
                                                        ["Bandwidth"], ["Bandwidth Deviation"], ["Zero Crossing Rate"], ["Zero Crossing Rate Deviations"], ["Decay Time"], ["Decay Time Deviation"], ["Tuning"], ["Tuning Deviation"]


    allAttackSpectrums, allSustainSpectrums, allDecaySpectrums, allAttackFrequencies, allSustainFrequencies, \
            allDecayFrequencies, seriesNames = [], [], [], [], [], [], []
    impulseTime, maxAttack, maxSustain, maxDecay = 0, 0, 0, 0

    # Flags used to decide which parameters will be calculated
    centroid_flag, rolloff_flag, rms_flag, bandwidth_flag, crossingRate_flag, tuning_flag = True, True, True, True, True, True

    # Calculating spectrums and parameters
    for seriesDirectory in os.listdir(os.fsencode(inputDirectory)):
        seriesDirectory = inputDirectory + "/" + os.fsdecode(seriesDirectory)
        print("Entering folder: " + seriesDirectory)
        impulses, attackSpectrums, sustainSpectrums, decaySpectrums, centroids, rolloffs, rmss, bandwidths, crossingRates, decayTimes, tunings = [], [], [], [], [], [], [], [], [], [], []

        for impulseFile in os.listdir(os.fsencode(seriesDirectory)):
            impulseFileName = seriesDirectory + "/" + os.fsdecode(impulseFile)
            print("Loading file: " + impulseFileName)
            impulse, samplingRate = librosa.load(impulseFileName)

            if centroid_flag:
                centroids.append(np.mean(librosa.feature.spectral_centroid(impulse)))
            if rolloff_flag:
                rolloffs.append(np.mean(librosa.feature.spectral_rolloff(impulse)))
            if bandwidth_flag:
                bandwidths.append(np.mean(librosa.feature.spectral_bandwidth(impulse)))
            if crossingRate_flag:
                crossingRates.append(np.mean(librosa.feature.zero_crossing_rate(impulse)))
            if rms_flag:
                rmss.append(CalculateRMS(impulse))
            if tuning_flag:
                tunings.append(np.mean(librosa.estimate_tuning(impulse)))
            if decayTime_flag:
                decayTimes.append(CalculateDecayTime(impulse, samplingRate))

            impulses = InsertIntoVstack(impulse, impulses)

        attackFrequencies, attackSpectrums, sustainFrequencies, sustainSpectrums, decayFrequencies, decaySpectrums = \
            CalculateFFTs(impulses, samplingRate, attackTime, sustainTime)

        avrAttackSpectrum = CalculateAverageVector(attackSpectrums)
        avrSustainSpectrum = CalculateAverageVector(sustainSpectrums)
        avrDecaySpectrum = CalculateAverageVector(decaySpectrums)

        allAttackSpectrums.append(avrAttackSpectrum)
        allSustainSpectrums.append(avrSustainSpectrum)
        allDecaySpectrums.append(avrDecaySpectrum)
        allAttackFrequencies.append(attackFrequencies)
        allSustainFrequencies.append(sustainFrequencies)
        allDecayFrequencies.append(decayFrequencies)
        seriesNames.append(seriesDirectory)
        impulseTime = len(impulses[0,:])/samplingRate
        maxAttack = max([maxAttack, max(avrAttackSpectrum)])
        maxSustain = max([maxSustain, max(avrSustainSpectrum)])
        maxDecay = max([maxDecay, max(avrDecaySpectrum)])

        seriesName = seriesDirectory.replace(inputDirectory + "/", "")
        series_names.append(seriesName)

        centroid_values.append(np.mean(centroids))
        centroid_deviations.append(np.std(centroids))
        rolloff_values.append(np.mean(rolloffs))
        rolloff_deviations.append(np.std(rolloffs))
        bandwidth_values.append(np.mean(bandwidths))
        bandwidth_deviation.append((np.std(bandwidths)))
        zeroCrossingRate_values.append(np.mean(crossingRates))
        zeroCrossingRate_deviations.append(np.std(crossingRates))
        rms_values.append(np.mean(rmss))
        rms_deviations.append(np.std(rmss))
        decayTime_values.append(np.mean(decayTimes))
        decayTime_deviations.append(np.std(decayTimes))
        tuning_values.append(np.mean(tunings))
        tuning_deviations.append(np.std(tunings))

    # Saving parameter data

    data_array = series_names
    if centroid_flag:
        data_array = np.vstack((data_array, centroid_values, centroid_deviations))
    if rolloff_flag:
        data_array = np.vstack((data_array, rolloff_values, rolloff_deviations))
    if rms_flag:
        data_array = np.vstack((data_array, rms_values, rms_deviations))
    if bandwidth_flag:
        data_array = np.vstack((data_array, bandwidth_values, bandwidth_deviation))
    if crossingRate_flag:
        data_array = np.vstack((data_array, zeroCrossingRate_values, zeroCrossingRate_values))
    if decayTime_flag:
        data_array = np.vstack((data_array, decayTime_values, decayTime_deviations))
    if tuning_flag:
        data_array = np.vstack((data_array, tuning_values, tuning_deviations))
    np.save(outputDirectory + '/' + parameterFileName + '_' + fileNameAppendix + '.npy', data_array)

    with open(outputDirectory + '/' + parameterFileName + '_' + fileNameAppendix + '.csv', 'w', newline='') as csvfile:
        dataWriter = csv.writer(csvfile, delimiter=',', quotechar=';', quoting=csv.QUOTE_MINIMAL)
        dataWriter.writerow(series_names)
        if centroid_flag:
            dataWriter.writerow(centroid_values)
            dataWriter.writerow(centroid_deviations)
        if rms_flag:
            dataWriter.writerow(rms_values)
            dataWriter.writerow(rms_deviations)
        if rolloff_flag:
            dataWriter.writerow(rolloff_values)
            dataWriter.writerow(rolloff_deviations)
        if bandwidth_flag:
            dataWriter.writerow(bandwidth_values)
            dataWriter.writerow(bandwidth_deviation)
        if crossingRate_flag:
            dataWriter.writerow(zeroCrossingRate_values)
            dataWriter.writerow(zeroCrossingRate_deviations)
        if decayTime_flag:
            dataWriter.writerow(decayTime_values)
            dataWriter.writerow(decayTime_deviations)
        if tuning_flag:
            dataWriter.writerow(tuning_values)
            dataWriter.writerow(tuning_deviations)

    print("Data saved to: " + parameterFileName + '_' + fileNameAppendix)

    # Saving spectrum data

    with open(outputDirectory + '/' + spectrumFileName + '_' + fileNameAppendix + '.csv', 'w', newline='') as csvfile:
        dataWriter = csv.writer(csvfile, delimiter=',', quotechar=';', quoting=csv.QUOTE_MINIMAL)
        for iterator in range(0, len(seriesNames)):
            dataWriter.writerow("Name: ")
            dataWriter.writerow(seriesNames[iterator])
            dataWriter.writerow("Attack spectrum: ")
            dataWriter.writerow(allAttackSpectrums[iterator])
            dataWriter.writerow("Attack frequencies: ")
            dataWriter.writerow(allAttackFrequencies[iterator])
            dataWriter.writerow("Sustain spectrum: ")
            dataWriter.writerow(allSustainSpectrums[iterator])
            dataWriter.writerow("Sustain frequencies: ")
            dataWriter.writerow(allSustainFrequencies[iterator])
            dataWriter.writerow("Decay Spectrum: ")
            dataWriter.writerow(allDecaySpectrums[iterator])
            dataWriter.writerow("Decay frequencies: ")
            dataWriter.writerow(allDecayFrequencies[iterator])

    print("Spectrums saved to: " + spectrumFileName + '_' + fileNameAppendix)

    # Drawing spectrum plots

    for iterator in range(0, len(seriesNames)):

        #plt.suptitle(seriesNames[iterator].replace(inputDirectory, ''), fontsize='xx-large')

        #converting frequencies to kHz for easier legibility
        kAttackFrequencies = allAttackFrequencies[iterator]/1000
        kSustainFrequencies = allSustainFrequencies[iterator]/1000
        kDecayFrequencies = allDecayFrequencies[iterator]/100

        plt.subplot(131)
        plt.plot(kAttackFrequencies, allAttackSpectrums[iterator], color = 'k')
        plt.locator_params(nbins=4)
        plt.xlim([0, 2])
        plt.ylim([10e-12, maxAttack * 1.1])
        plt.xlabel('f [kHz]', fontsize=30)
        plt.xticks(fontsize=21)
        plt.yticks(fontsize=23)

        #Mierna próba zmiany wielkości notacji wykładniczej
        #yaxis.get_offset_text().set_fontsize(24)

        plt.ticklabel_format(useMathText=True, scilimits=(0, 0))
        plt.ylabel('Amplitude', fontsize=35)
        plt.title('0 - ' + str(attackTime) + ' [s]', fontsize=25)
        #plt.yscale("log")
        #plt.text(1500, maxAttack, str('Energy = ' + str(CalculateRMS(allAttackSpectrums[iterator]))))

        plt.subplot(132)
        plt.plot(kSustainFrequencies, allSustainSpectrums[iterator], color = 'k')
        plt.locator_params(nbins=4)
        plt.xlim([0, 2])
        plt.ylim([10e-12, maxSustain * 1.1])
        plt.xlabel('f [kHz]', fontsize=32)
        plt.xticks(fontsize=21)
        plt.ticklabel_format(useMathText=True, scilimits=(0, 0))
        plt.yticks(fontsize=23)
        plt.title(str(attackTime) + ' - ' + str(sustainTime) + ' [s]', fontsize=25)
        #plt.yscale("log")
        #plt.text(1500, maxSustain, str('Energy = ' + str(CalculateRMS(allSustainSpectrums[iterator]))))

        plt.subplot(133)
        plt.plot(kDecayFrequencies, allDecaySpectrums[iterator], color = 'k')
        plt.locator_params(nbins=4)
        plt.xlim([0, 2])
        plt.ylim([10e-12, maxDecay * 1.1])
        plt.xlabel('f [kHz]', fontsize=32)
        plt.xticks(fontsize=21)
        plt.ticklabel_format(useMathText=True, scilimits=(0, 0))
        plt.yticks(fontsize=23)
        plt.title(str(sustainTime) + ' - ' + str(round(impulseTime, 2)) + ' [s]', fontsize=25)
        #plt.yscale("log")
        #plt.text(1500, maxDecay, str('Energy = ' + str(CalculateRMS(allDecaySpectrums[iterator]))))


        outputFile = seriesNames[iterator].replace(inputDirectory, outputDirectory)
        print("Outputing to: " + outputFile)

        figure = plt.gcf()
        figure.set_size_inches(19, 8)
        plt.savefig(outputFile, dpi = 100)

        #This should be used if outputing to vector file
        #plt.savefig(outputFile, dpi=100, format="eps")

        #plt.show()
        plt.clf()








