import numpy as np
from scipy import signal
import os
import matplotlib.pyplot as plt
import librosa
import librosa.display
import iracema
import shutil
import csv
import ParameterCalculator as pc

# This script accepts folders with individually parsed impulses and calculates an averaged spectrum for each directory.
# The plots of these results are saved into the output directory.
# Spectral and energy parameters are also calculated and saved into the output directory.

def CalculateStatistics(values, meanValues, deviations):
    meanValues.append(np.mean(values))
    deviations.append(np.std(values))
    return 0

def run(inputDirectory, outputDirectory, parameterFileName, spectrumFileName, fileNameAppendix, attackTime, sustainTime, decayTime_flag):

    if os.path.isdir(outputDirectory):
            shutil.rmtree(outputDirectory)
    os.mkdir(outputDirectory)
    samplingRate = 0

    # These variables are used to save the parameter data
    data_array = []
    series_names, centroid_values, centroid_deviations,  f0NormalizedCentroid_values, f0NormalizedCentroid_deviations, \
    rolloff_values, rolloff_deviations, bandwidth_values, bandwidth_deviation, spread_values, spread_deviations, \
    highLowEnergy_values, highLowEnegry_deviations, tristimulus1_values, tristimulus1_deviations, \
    tristimulus2_values, tristimulus2_deviations, tristimulus3_values, tristimulus3_deviations, \
    inharmonicity_values, inharmonicity_deviations, noisiness_values, noisiness_deviations, \
    oddEvenRatio_values, oddEvenRatio_deviations, tuning_values, tuning_deviations, \
    zeroCrossingRate_values, zeroCrossingRate_deviations, rms_values, rms_deviations, entropy_values, entropy_deviations, \
    temporalCentroid_values, temporalCentroid_deviations, logAttackTime_values, logAttackTime_deviations, \
    decayTime_values, decayTime_deviations, foundFundumentalPitches = \
    [" "], ["Spectrum Centroid"], ["Centroid Deviation"], ["F0 Normalized Centroid"], ["F0 Normalized Centroid Deviations"],\
    ["Rolloff"], ["Rolloff Deviation"], ["Bandwidth"], ["Bandwidth Deviation"], ["Spread"], ["Spread Deviation"], \
    ["High Energy - Low Energy Ratio"], ["High Energy - Low Energy Ratio Deviations"], ["Tristimulus 1"], ["Tristimulus 1 Deviations"], \
    ["Tristimulus 2"], ["Tristimulus 2 Deviations"], ["Tristimulus 3"], ["Tristimulus 3 Deviations"], \
    ["Inharmonicity"], ["Inharmonicity Deviation"], ["Noisiness"], ["Noisiness Deviations"], \
    ["Odd-Even Ratio"], ["Odd-Even Ratio Deviation"], ["Tuning"], ["Tuning Deviation"], \
    ["Zero Crossing Rate"], ["Zero Crossing Rate Deviation"], ["RMS"], ["RMS Deviation"], ["Entropy"], ["Entropy Deviation"], \
    ["Temporal Centroid"], ["Temporal Centroid Deviations"], ["Log Attack Time"], ["Log Attack Time Deviations"], \
    ["Decay Time"], ["Decay Time Deviation"], ["Average Found Fundumental Pitches"]

    allAttackSpectrums, allSustainSpectrums, allDecaySpectrums, allAttackFrequencies, allSustainFrequencies, \
            allDecayFrequencies, seriesNames = [], [], [], [], [], [], []
    # Spectrum scaling factors
    impulseTime, maxAttack, maxSustain, maxDecay = 0, 0, 0, 0

    # Flags used to decide which parameters will be calculated
    centroid_flag, f0normCentroid_flag, rolloff_flag, bandwidth_flag, spread_flag, highLowEnergy_flag, tristimulus_flag, \
    inharmonicity_flag, noisiness_flag, oddeven_flag, tuning_flag, crossingRate_flag, rms_flag, entropy_flag, \
    temporalCentroid_flag, logAttackTime_flag = True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True

    # Calculating spectrums and parameters
    for seriesDirectory in os.listdir(os.fsencode(inputDirectory)):
        seriesDirectory = inputDirectory + "/" + os.fsdecode(seriesDirectory)
        print("Entering folder: " + seriesDirectory)
        impulses, attackSpectrums, sustainSpectrums, decaySpectrums, centroids, f0normCentroids, rolloffs, bandwidths, \
        spreads, highLowEnergies, tristimulus1s, tristimulus2s, tristimulus3s, inharmonicities, noisinesses, \
        oddEvenRatios, tunings, crossingRates, rmss, entropies, temporalCentroids, logAttackTimes, decayTimes, \
        pitchesHz =  [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []

        # If harmonic data and normalized centroids make no sense it may be caused by improper fundumental pitch detection.
        # Check in parameterData.csv whether the fundumentals were properly found.
        # If not, then manually add the correct pitch below and rerun the offending sounds.
        fundumentalPitch = 0

        for impulseFile in os.listdir(os.fsencode(seriesDirectory)):
            impulseFileName = seriesDirectory + "/" + os.fsdecode(impulseFile)
            # librosa loading
            print("Loading file: " + impulseFileName)
            impulse, samplingRate = librosa.load(impulseFileName)
            # iracema loading
            impulseIRA = iracema.Audio(impulseFileName)
            impulseFFT = iracema.spectral.fft(impulseIRA, window_size=2048, hop_size=1024)
            pitch = iracema.pitch.hps(impulseFFT, minf0=50, maxf0=500)
            pitchesHz.append(np.median(pitch.data))
            harmonics = iracema.harmonics.extract(impulseFFT, pitch)

            if centroid_flag:
                centroids.append(np.mean(librosa.feature.spectral_centroid(impulse)))
            if f0normCentroid_flag:
                if fundumentalPitch == 0:
                    f0normCentroids.append((np.mean(librosa.feature.spectral_centroid(impulse) / np.median(pitch.data))))
                else:
                    f0normCentroids.append((np.mean(librosa.feature.spectral_centroid(impulse) / fundumentalPitch)))
            if rolloff_flag:
                rolloffs.append(np.mean(librosa.feature.spectral_rolloff(impulse)))
            if bandwidth_flag:
                bandwidths.append(np.mean(librosa.feature.spectral_bandwidth(impulse)))
            if spread_flag:
                spreads.append(np.mean(iracema.features.spectral_spread(impulseFFT).data))
            if noisiness_flag:
                noisinesses.append(np.mean(iracema.features.noisiness(impulseFFT, harmonics['magnitude']).data))
            if tuning_flag:
                tunings.append(np.mean(librosa.estimate_tuning(impulse)))
            if crossingRate_flag:
                crossingRates.append(np.mean(librosa.feature.zero_crossing_rate(impulse)))
            if rms_flag:
                rmss.append(pc.CalculateRMS(impulse))
            if entropy_flag:
                entropies.append(np.mean(iracema.features.spectral_entropy(impulseFFT).data))
            if temporalCentroid_flag:
                temporalCentroids.append(pc.CalculateTemporalCentroid(impulseIRA))
            if logAttackTime_flag:
                logAttackTimes.append(pc.CalculateLogAttackTime(impulseIRA))
            if decayTime_flag:
                decayTimes.append(pc.CalculateDecayTime(impulseIRA))
            impulses = pc.InsertIntoVstack(impulse, impulses)

        fullSpectrums, fullFrequencies, attackFrequencies, attackSpectrums, sustainFrequencies, sustainSpectrums, \
        decayFrequencies, decaySpectrums = pc.CalculateFFTs(impulses, samplingRate, attackTime, sustainTime)

        if fundumentalPitch == 0:
            fundumentalPitch = np.median(pitchesHz)
        foundFundumentalPitches.append(fundumentalPitch)
        mathHarmFreq = pc.CreateMathematicalHarmonicFrequencyVector(fundumentalPitch, n=15)
        harmonicData = pc.ExtractHarmonicDataFromSpectrums(fullSpectrums, fullFrequencies, mathHarmFreq, bufforInHZ=20)

        if highLowEnergy_flag:
            highLowEnergies = pc.CalculateHighEnergyLowEnergyRatio(fullSpectrums, fullFrequencies)
        if tristimulus_flag:
            tristimulus1s, tristimulus2s, tristimulus3s = pc.CalculateTristimulus(harmonicData)
        if inharmonicity_flag:
            inharmonicities = pc.CalculateInharmonicity(harmonicData)
        if oddeven_flag:
            oddEvenRatios = pc.CalculateOERs(harmonicData)

        avrAttackSpectrum = pc.CalculateAverageVector(attackSpectrums)
        avrSustainSpectrum = pc.CalculateAverageVector(sustainSpectrums)
        avrDecaySpectrum = pc.CalculateAverageVector(decaySpectrums)

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

        CalculateStatistics(centroids, centroid_values, centroid_deviations)
        CalculateStatistics(f0normCentroids, f0NormalizedCentroid_values, f0NormalizedCentroid_deviations)
        CalculateStatistics(rolloffs, rolloff_values, rolloff_deviations)
        CalculateStatistics(bandwidths, bandwidth_values, bandwidth_deviation)
        CalculateStatistics(spreads, spread_values, spread_deviations)
        CalculateStatistics(highLowEnergies, highLowEnergy_values, highLowEnegry_deviations)
        CalculateStatistics(tristimulus1s, tristimulus1_values, tristimulus1_deviations)
        CalculateStatistics(tristimulus2s, tristimulus2_values, tristimulus2_deviations)
        CalculateStatistics(tristimulus3s, tristimulus3_values, tristimulus3_deviations)
        CalculateStatistics(inharmonicities, inharmonicity_values, inharmonicity_deviations)
        CalculateStatistics(noisinesses, noisiness_values, noisiness_deviations)
        CalculateStatistics(oddEvenRatios, oddEvenRatio_values, oddEvenRatio_deviations)
        CalculateStatistics(tunings, tuning_values, tuning_deviations)
        CalculateStatistics(crossingRates, zeroCrossingRate_values, zeroCrossingRate_deviations)
        CalculateStatistics(rmss, rms_values, rms_deviations)
        CalculateStatistics(entropies, entropy_values, entropy_deviations)
        CalculateStatistics(temporalCentroids, temporalCentroid_values, temporalCentroid_deviations)
        CalculateStatistics(logAttackTimes, logAttackTime_values, logAttackTime_deviations)
        CalculateStatistics(decayTimes, decayTime_values, decayTime_deviations)


    # Saving parameter data into .npy file
    data_array = series_names
    if centroid_flag:
        data_array = np.vstack((data_array, centroid_values, centroid_deviations))
    if f0normCentroid_flag:
        data_array = np.vstack((data_array, f0NormalizedCentroid_values, f0NormalizedCentroid_deviations))
    if rolloff_flag:
        data_array = np.vstack((data_array, rolloff_values, rolloff_deviations))
    if bandwidth_flag:
        data_array = np.vstack((data_array, bandwidth_values, bandwidth_deviation))
    if spread_flag:
        data_array = np.vstack((data_array, spread_values, spread_deviations))
    if highLowEnergy_flag:
        data_array = np.vstack((data_array, highLowEnergy_values, highLowEnegry_deviations))
    if tristimulus_flag:
        data_array = np.vstack((data_array, tristimulus1_values, tristimulus1_deviations))
        data_array = np.vstack((data_array, tristimulus2_values, tristimulus2_deviations))
        data_array = np.vstack((data_array, tristimulus3_values, tristimulus3_deviations))
    if inharmonicity_flag:
        data_array = np.vstack((data_array, inharmonicity_values, inharmonicity_deviations))
    if noisiness_flag:
        data_array = np.vstack((data_array, noisiness_values, noisiness_deviations))
    if oddeven_flag:
        data_array = np.vstack((data_array, oddEvenRatio_values, oddEvenRatio_deviations))
    if tuning_flag:
        data_array = np.vstack((data_array, tuning_values, tuning_deviations))
    if crossingRate_flag:
        data_array = np.vstack((data_array, zeroCrossingRate_values, zeroCrossingRate_deviations))
    if rms_flag:
        data_array = np.vstack((data_array, rms_values, rms_deviations))
    if entropy_flag:
        data_array = np.vstack((data_array, entropy_values, entropy_deviations))
    if temporalCentroid_flag:
        data_array = np.vstack((data_array, temporalCentroid_values, temporalCentroid_deviations))
    if logAttackTime_flag:
        data_array = np.vstack((data_array, logAttackTime_values, logAttackTime_deviations))
    if decayTime_flag:
        data_array = np.vstack((data_array, decayTime_values, decayTime_deviations))

    np.save(outputDirectory + '/' + parameterFileName + '_' + fileNameAppendix + '.npy', data_array)

    # saving data into .csv file
    with open(outputDirectory + '/' + parameterFileName + '_' + fileNameAppendix + '.csv', 'w', newline='') as csvfile:
        dataWriter = csv.writer(csvfile, delimiter=',', quotechar=';', quoting=csv.QUOTE_MINIMAL)
        dataWriter.writerow(series_names)
        if centroid_flag:
            dataWriter.writerow(centroid_values)
            dataWriter.writerow(centroid_deviations)
        if f0normCentroid_flag:
            dataWriter.writerow(f0NormalizedCentroid_values)
            dataWriter.writerow(f0NormalizedCentroid_deviations)
        if rolloff_flag:
            dataWriter.writerow(rolloff_values)
            dataWriter.writerow(rolloff_deviations)
        if bandwidth_flag:
            dataWriter.writerow(bandwidth_values)
            dataWriter.writerow(bandwidth_deviation)
        if spread_flag:
            dataWriter.writerow(spread_values)
            dataWriter.writerow(spread_deviations)
        if highLowEnergy_flag:
            dataWriter.writerow(highLowEnergy_values)
            dataWriter.writerow(highLowEnegry_deviations)
        if tristimulus_flag:
            dataWriter.writerow(tristimulus1_values)
            dataWriter.writerow(tristimulus1_deviations)
            dataWriter.writerow(tristimulus2_values)
            dataWriter.writerow(tristimulus2_deviations)
            dataWriter.writerow(tristimulus3_values)
            dataWriter.writerow(tristimulus3_deviations)
        if inharmonicity_flag:
            dataWriter.writerow(inharmonicity_values)
            dataWriter.writerow(inharmonicity_deviations)
        if noisiness_flag:
            dataWriter.writerow(noisiness_values)
            dataWriter.writerow(noisiness_deviations)
        if oddeven_flag:
            dataWriter.writerow(oddEvenRatio_values)
            dataWriter.writerow(oddEvenRatio_deviations)
        if tuning_flag:
            dataWriter.writerow(tuning_values)
            dataWriter.writerow(tuning_deviations)
        if crossingRate_flag:
            dataWriter.writerow(zeroCrossingRate_values)
            dataWriter.writerow(zeroCrossingRate_deviations)
        if rms_flag:
            dataWriter.writerow(rms_values)
            dataWriter.writerow(rms_deviations)
        if entropy_flag:
            dataWriter.writerow(entropy_values)
            dataWriter.writerow(entropy_deviations)
        if temporalCentroid_flag:
            dataWriter.writerow(temporalCentroid_values)
            dataWriter.writerow((temporalCentroid_deviations))
        if logAttackTime_flag:
            dataWriter.writerow(logAttackTime_values)
            dataWriter.writerow(logAttackTime_deviations)
        if decayTime_flag:
            dataWriter.writerow(decayTime_values)
            dataWriter.writerow(decayTime_deviations)
        dataWriter.writerow(foundFundumentalPitches)

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








