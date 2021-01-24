import numpy as np
from scipy import signal
import os
import matplotlib.pyplot as plt
import librosa
import librosa.display
import iracema
import shutil
import csv

# This script accepts folders with individually parsed impulses and calculates an averaged spectrum for each directory.
# The plots of these results are saved into the output directory.
# Spectral and energy parameters are also calculated and saved into the output directory.

class Harmonics:
    def __init__(self, frequencies, amplitudes):
        self.frequencies = frequencies
        self.amplitudes = amplitudes

def InsertIntoVstack(vector, stack):
    if stack == []:  # Not very elegant way to make sure the first impulse is loaded in correctly
        stack = vector
    else:
        stack = np.vstack([stack, vector])

    return stack

def CalculateAverageVector(Vectors): # Takes a vector of vectors and calculates a average vector
    averageVector = np.mean(Vectors, axis=0)
    return averageVector

def CalculateFFTs(takes, samplingRate, attackTime, sustainTime): # Takes array of impulses and creates array of spectrums
    attackSpectrums, sustainSpectrums, decaySpectrums, fullSpectrums = [], [], [], []
    attackFrequencies, sustainFrequencies, decayFrequencies, fullFrequencies = 0, 0, 0, 0
    for take in takes:
        fullFrequencies, fullSpectrum = signal.periodogram(take, samplingRate, scaling="spectrum")
        attackFrequencies, attackSpectrum = signal.periodogram(take[:int(attackTime*samplingRate)], samplingRate, scaling="spectrum")
        sustainFrequencies, sustainSpectrum = signal.periodogram(take[int(attackTime*samplingRate):int(sustainTime*samplingRate)], samplingRate, scaling="spectrum")
        decayFrequencies, decaySpectrum = signal.periodogram(take[int(sustainTime*samplingRate):], samplingRate, scaling="spectrum")

        fullSpectrums = InsertIntoVstack(fullSpectrum.real, fullSpectrums)
        attackSpectrums = InsertIntoVstack(attackSpectrum.real, attackSpectrums)
        sustainSpectrums = InsertIntoVstack(sustainSpectrum.real, sustainSpectrums)
        decaySpectrums = InsertIntoVstack(decaySpectrum.real, decaySpectrums)

    return fullSpectrums, fullFrequencies, attackFrequencies, attackSpectrums, sustainFrequencies, sustainSpectrums, decayFrequencies, decaySpectrums

def CalculateRMS(signal):
    return np.sum(librosa.feature.rmse(signal))

def CalculateDecayTime(impulse, windowLength = 2048, hopsize = 1024, ratio = 0.12): # Calculates time between the peak of impulse and it decaying below the value of max*ratio
    envelope = iracema.features.peak_envelope(impulse, windowLength, hopsize)
    maxEnv = max(envelope.data)
    peakTime = 0
    decayTime = 0

    for i in range(0,len(envelope.data)):
        if envelope.data[i] == maxEnv:
            peakTime = envelope.time[i]
        if maxEnv * ratio > envelope.data[i]:
            decayTime = envelope.time[i] - peakTime
            break
    return decayTime

def CalculateOERs(harmonicsData):
    oers = []
    for take in harmonicsData:
        odd, even = 0, 0
        for i in range (0, len(take.amplitudes)):
            if (i+1)%2 == 1:
                odd = odd + pow(take.amplitudes[i], 2)
            else:
                even = even + pow(take.amplitudes[i], 2)
        oers.append(odd/even)
    return oers

def CalculateTristimulus(harmonicsData):
    tristimulus1s, tristimulus2s, tristimulus3s = [], [], []
    for take in harmonicsData:
        allAmplitudes = 0
        fiveUpAmplitudes = 0
        for i in range (0, len(take.amplitudes)):
            allAmplitudes = allAmplitudes + take.amplitudes[i]
            if i >= 5:
                fiveUpAmplitudes = fiveUpAmplitudes + take.amplitudes[i]
        tristimulus1s.append(take.amplitudes[0]/allAmplitudes)
        tristimulus2s.append((take.amplitudes[1] + take.amplitudes[2] + take.amplitudes[3])/allAmplitudes)
        tristimulus3s.append(fiveUpAmplitudes/allAmplitudes)
    return tristimulus1s, tristimulus2s, tristimulus3s

def CreateMathematicalHarmonicFrequencyVector(pitch, n):
    freq = []
    for i in range (1, n):
        freq.append(pitch * i)
    return freq

def ExtractHarmonicDataFromSpectrums(spectrums, spectrumFrequencies, mathHarmonicFrequencies, bufforInHZ = 5):
    buffor = int(bufforInHZ/(spectrumFrequencies[1]-spectrumFrequencies[0]))
    harmonicData = []

    for spectrum in spectrums:
        amplitudes = []
        harmonicFrequencies = []
        harmonicNumber = 0

        for fftSample in range (0, len(spectrum)):
            if spectrumFrequencies[fftSample] > mathHarmonicFrequencies[harmonicNumber]:
                peakValue = 0
                peakFrequency = 0

                for i in range (fftSample-buffor, fftSample+buffor):
                    if spectrum[i] > peakValue:
                        peakValue = spectrum[i]
                        peakFrequency = spectrumFrequencies[i]

                amplitudes.append(peakValue)
                harmonicFrequencies.append(peakFrequency)

                if harmonicNumber < len(mathHarmonicFrequencies) - 1:
                    harmonicNumber = harmonicNumber + 1
                else:
                    break

        # showing found harmonics and spectrum for debugging
        if False:
            x = []
            for y in range(0, len(harmonicFrequencies)):
                x.append(y)
            plt.subplot(121)
            plt.plot(spectrumFrequencies, spectrum)
            plt.subplot(122)
            plt.bar(x, amplitudes)
            plt.show()

        harmonicData.append(Harmonics(harmonicFrequencies, amplitudes))
    return harmonicData

def run(inputDirectory, outputDirectory, parameterFileName, spectrumFileName, fileNameAppendix, attackTime, sustainTime, decayTime_flag):

    if os.path.isdir(outputDirectory):
            shutil.rmtree(outputDirectory)
    os.mkdir(outputDirectory)

    samplingRate = 0
    # These variables are used to save the parameter data to a csv file
    data_array = []
    series_names, centroid_values, centroid_deviations, rolloff_values, rolloff_deviations, rms_values, rms_deviations, \
        bandwidth_values, bandwidth_deviation, zeroCrossingRate_values, zeroCrossingRate_deviations, spread_values, spread_deviations, \
        entropy_values, entropy_deviations, inharmonicity_values, inharmonicity_deviations, noisiness_values, noisiness_deviations, \
        oddEvenRatio_values, oddEvenRatio_deviations, tristimulus1_values, tristimulus1_deviations, tristimulus2_values, tristimulus2_deviations, \
        tristimulus3_values, tristimulus3_deviations, decayTime_values, decayTime_deviations, tuning_values, tuning_deviations = [" "], ["Spectrum Centroid"], ["Centroid Deviation"], \
                ["Rolloff"], ["Rolloff Deviation"], ["RMS"], ["RMS Deviation"], ["Bandwidth"], ["Bandwidth Deviation"], ["Zero Crossing Rate"], \
            ["Zero Crossing Rate Deviation"], ["Spread"], ["Spread Deviation"], ["Entropy"], ["Entropy Deviation"], ["Inharmonicity"], \
            ["Inharmonicity Deaviation"], ["Noisiness"], ["Noisiness Deviations"], ["Odd-Even Ratio"], ["Odd-Even Ratio Deviation"], \
            ["Tristimulus 1"], ["Tristimulus 1 Deviations"], ["Tristimulus 2"], ["Tristimulus 2 Deviations"], ["Tristimulus 3"], ["Tristimulus 3 Deviations"], \
            ["Decay Time"], ["Decay Time Deviation"], ["Tuning"], ["Tuning Deviation"]


    allAttackSpectrums, allSustainSpectrums, allDecaySpectrums, allAttackFrequencies, allSustainFrequencies, \
            allDecayFrequencies, seriesNames = [], [], [], [], [], [], []
    impulseTime, maxAttack, maxSustain, maxDecay = 0, 0, 0, 0

    # Flags used to decide which parameters will be calculated
    centroid_flag, rolloff_flag, rms_flag, bandwidth_flag, crossingRate_flag, spread_flag, entropy_flag, inharmonicity_flag, \
    noisiness_flag, oddeven_flag, tristimulus_flag, tuning_flag = True, True, True, True, True, True, True, True, True, True, True, True

    # Calculating spectrums and parameters
    for seriesDirectory in os.listdir(os.fsencode(inputDirectory)):
        seriesDirectory = inputDirectory + "/" + os.fsdecode(seriesDirectory)
        print("Entering folder: " + seriesDirectory)
        impulses, attackSpectrums, sustainSpectrums, decaySpectrums, centroids, rolloffs, rmss, bandwidths, \
        crossingRates, spreads, entropies, inharmonicities, noisinesses, oddEvenRatios, tristimulus1s, \
        tristimulus2s, tristimulus3s, decayTimes, tunings = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
        pitchHz = 0

        for impulseFile in os.listdir(os.fsencode(seriesDirectory)):
            impulseFileName = seriesDirectory + "/" + os.fsdecode(impulseFile)
            # librosa loading
            print("Loading file: " + impulseFileName)
            impulse, samplingRate = librosa.load(impulseFileName)
            # iracema loading
            impulseIRA = iracema.Audio(impulseFileName)
            impulseFFT = iracema.spectral.fft(impulseIRA, window_size=2048, hop_size=1024)
            pitch = iracema.pitch.hps(impulseFFT, minf0=50, maxf0=500)
            pitchHz = np.median(pitch.data)
            harmonics = iracema.harmonics.extract(impulseFFT, pitch)

            if centroid_flag:
                centroids.append(np.mean(librosa.feature.spectral_centroid(impulse)))
            if rolloff_flag:
                rolloffs.append(np.mean(librosa.feature.spectral_rolloff(impulse)))
            if bandwidth_flag:
                bandwidths.append(np.mean(librosa.feature.spectral_bandwidth(impulse)))
            if crossingRate_flag:
                crossingRates.append(np.mean(librosa.feature.zero_crossing_rate(impulse)))
            if spread_flag:
                spreads.append(np.mean(iracema.features.spectral_centroid(impulseFFT).data))
            if entropy_flag:
                entropies.append(np.mean(iracema.features.spectral_entropy(impulseFFT).data))
            #if inharmonicity_flag:
            #    inharmonicities.append(np.mean(iracema.features.inharmonicity(impulseFFT, harmonics)))
            if noisiness_flag:
                noisinesses.append(np.mean(iracema.features.noisiness(impulseFFT, harmonics['magnitude']).data))
            if rms_flag:
                rmss.append(CalculateRMS(impulse))
            if tuning_flag:
                tunings.append(np.mean(librosa.estimate_tuning(impulse)))
            if decayTime_flag:
                decayTimes.append(CalculateDecayTime(impulseIRA))

            impulses = InsertIntoVstack(impulse, impulses)

        fullSpectrums, fullFrequencies, attackFrequencies, attackSpectrums, sustainFrequencies, sustainSpectrums, decayFrequencies, decaySpectrums = \
            CalculateFFTs(impulses, samplingRate, attackTime, sustainTime)

        mathHarmFreq = CreateMathematicalHarmonicFrequencyVector(pitchHz, n=15)
        harmonicData = ExtractHarmonicDataFromSpectrums(fullSpectrums, fullFrequencies, mathHarmFreq, bufforInHZ=20)
        if oddeven_flag:
            oddEvenRatios = CalculateOERs(harmonicData)
        if tristimulus_flag:
            tristimulus1s, tristimulus2s, tristimulus3s = CalculateTristimulus(harmonicData)

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
        spread_values.append(np.mean(spreads))
        spread_deviations.append(np.std(spreads))
        entropy_values.append(np.mean(entropies))
        entropy_deviations.append(np.std(entropies))
        #inharmonicity_values.append(np.mean(inharmonicities))
        #inharmonicity_deviations.append(np.std(inharmonicities))
        noisiness_values.append(np.mean(noisinesses))
        noisiness_deviations.append(np.std(noisinesses))
        oddEvenRatio_values.append(np.mean(oddEvenRatios))
        oddEvenRatio_deviations.append(np.std(oddEvenRatios))
        tristimulus1_values.append(np.mean(tristimulus1s))
        tristimulus1_deviations.append(np.std(tristimulus1s))
        tristimulus2_values.append(np.mean(tristimulus2s))
        tristimulus2_deviations.append(np.std(tristimulus2s))
        tristimulus3_values.append(np.mean(tristimulus3s))
        tristimulus3_deviations.append(np.std(tristimulus3s))
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
        data_array = np.vstack((data_array, zeroCrossingRate_values, zeroCrossingRate_deviations))
    if spread_flag:
        data_array = np.vstack((data_array, spread_values, spread_deviations))
    if entropy_flag:
        data_array = np.vstack((data_array, entropy_values, entropy_deviations))
    #if inharmonicity_flag:
    #    data_array = np.vstack((data_array, inharmonicity_values, inharmonicity_deviations))
    if noisiness_flag:
        data_array = np.vstack((data_array, noisiness_values, noisiness_deviations))
    if oddeven_flag:
        data_array = np.vstack((data_array, oddEvenRatio_values, oddEvenRatio_deviations))
    if tristimulus_flag:
        data_array = np.vstack((data_array, tristimulus1_values, tristimulus1_deviations))
        data_array = np.vstack((data_array, tristimulus2_values, tristimulus2_deviations))
        data_array = np.vstack((data_array, tristimulus3_values, tristimulus3_deviations))
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
        if spread_flag:
            dataWriter.writerow(spread_values)
            dataWriter.writerow(spread_deviations)
        if entropy_flag:
            dataWriter.writerow(entropy_values)
            dataWriter.writerow(entropy_deviations)
    #    if inharmonicity_flag:
    #        dataWriter.writerow(inharmonicity_values)
    #        dataWriter.writerow(inharmonicity_deviations)
        if noisiness_flag:
            dataWriter.writerow(noisiness_values)
            dataWriter.writerow(noisiness_deviations)
        if oddeven_flag:
            dataWriter.writerow(oddEvenRatio_values)
            dataWriter.writerow(oddEvenRatio_deviations)
        if tristimulus_flag:
            dataWriter.writerow(tristimulus1_values)
            dataWriter.writerow(tristimulus1_deviations)
            dataWriter.writerow(tristimulus2_values)
            dataWriter.writerow(tristimulus2_deviations)
            dataWriter.writerow(tristimulus3_values)
            dataWriter.writerow(tristimulus3_deviations)
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








