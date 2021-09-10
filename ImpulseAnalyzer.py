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
class Arguments:
    pass

def CalculateStatistics(values, meanValues, deviations):
    meanValues.append(np.mean(values))
    deviations.append(np.std(values))
    return 0

def run(inputDirectory, outputDirectory, parameterFileName, spectrumFileName, fileNameAppendix, attackTime, sustainTime, \
    centroid_flag, f0normCentroid_flag, rolloff_flag, bandwidth_flag, spread_flag, flux_flag, highLowEnergy_flag, \
    subBandFlux_flag, tristimulus_flag, inharmonicity_flag, noisiness_flag, oddeven_flag, tuning_flag, crossingRate_flag, \
    rms_flag, entropy_flag, temporalCentroid_flag, logAttackTime_flag, decayTime_flag, vectorOutput_flag):

    # clear output folder
    if os.path.isdir(outputDirectory):
            shutil.rmtree(outputDirectory)
    os.mkdir(outputDirectory)
    samplingRate = 0

    # ----------Setting up variables required for calculation and saving of parameter data---------------
    data_array = []
    series_names, centroid_values, centroid_deviations,  f0NormalizedCentroid_values, f0NormalizedCentroid_deviations, \
    rolloff_values, rolloff_deviations, bandwidth_values, bandwidth_deviation, spread_values, spread_deviations, \
    flux_values, flux_deviations, highLowEnergy_values, highLowEnegry_deviations, \
    subBandFlux1_values, subBandFlux1_deviations, subBandFlux2_values, subBandFlux2_deviations, \
    subBandFlux3_values, subBandFlux3_deviations, subBandFlux4_values, subBandFlux4_deviations, \
    subBandFlux5_values, subBandFlux5_deviations, subBandFlux6_values, subBandFlux6_deviations, \
    subBandFlux7_values, subBandFlux7_deviations, subBandFlux8_values, subBandFlux8_deviations, \
    subBandFlux9_values, subBandFlux9_deviations, subBandFlux10_values, subBandFlux10_deviations, \
    tristimulus1_values, tristimulus1_deviations, tristimulus2_values, tristimulus2_deviations, \
    tristimulus3_values, tristimulus3_deviations, \
    inharmonicity_values, inharmonicity_deviations, noisiness_values, noisiness_deviations, \
    oddEvenRatio_values, oddEvenRatio_deviations, tuning_values, tuning_deviations, \
    zeroCrossingRate_values, zeroCrossingRate_deviations, rms_values, rms_deviations, entropy_values, entropy_deviations, \
    temporalCentroid_values, temporalCentroid_deviations, logAttackTime_values, logAttackTime_deviations, \
    decayTime_values, decayTime_deviations, foundFundumentalPitches = \
    [" "], ["Spectrum Centroid"], ["Centroid Deviation"], ["F0 Normalized Centroid"], ["F0 Normalized Centroid Deviations"],\
    ["Rolloff"], ["Rolloff Deviation"], ["Bandwidth"], ["Bandwidth Deviation"], ["Spread"], ["Spread Deviation"], \
    ["Spectral Flux"], ["Spectral Flux Deviations"], ["High Energy - Low Energy Ratio"], ["High Energy - Low Energy Ratio Deviations"], \
    ["Sub-Band Flux 1"], ["Sub-Band Flux 1 Deviation"], ["Sub-Band Flux 2"], ["Sub-Band Flux 2 Deviation"], \
    ["Sub-Band Flux 3"], ["Sub-Band Flux 3 Deviation"], ["Sub-Band Flux 4"], ["Sub-Band Flux 4 Deviation"], \
    ["Sub-Band Flux 5"], ["Sub-Band Flux 5 Deviation"], ["Sub-Band Flux 6"], ["Sub-Band Flux 6 Deviation"], \
    ["Sub-Band Flux 7"], ["Sub-Band Flux 7 Deviation"], ["Sub-Band Flux 8"], ["Sub-Band Flux 8 Deviation"], \
    ["Sub-Band Flux 9"], ["Sub-Band Flux 9 Deviation"], ["Sub-Band Flux 10"], ["Sub-Band Flux 10 Deviation"], \
    ["Tristimulus 1"], ["Tristimulus 1 Deviations"], \
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

    mfcc_flag = False

    # ---------------Calculating spectrums and parameters------------------
    for seriesDirectory in os.listdir(os.fsencode(inputDirectory)):
        seriesDirectory = inputDirectory + "/" + os.fsdecode(seriesDirectory)
        print("Entering folder: " + seriesDirectory)
        impulses, attackSpectrums, sustainSpectrums, decaySpectrums, centroids, f0normCentroids, rolloffs, bandwidths, \
        spreads, fluxes, highLowEnergies, subBandFluxes1, subBandFluxes2, subBandFluxes3, subBandFluxes4, subBandFluxes5, \
        subBandFluxes6, subBandFluxes7, subBandFluxes8, subBandFluxes9, subBandFluxes10, \
        tristimulus1s, tristimulus2s, tristimulus3s, inharmonicities, noisinesses, \
        oddEvenRatios, tunings, crossingRates, rmss, entropies, temporalCentroids, logAttackTimes, decayTimes, \
        pitchesHz =  [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []

        # If harmonic data and normalized centroids make no sense it may be caused by improper fundumental pitch detection.
        # Check in parameterData.csv whether the fundumentals were properly found.
        # If not, then manually add the correct pitch below and rerun the offending sounds.
        fundumentalPitch = 0 #523.26 #261.63

        for impulseFile in os.listdir(os.fsencode(seriesDirectory)):
            impulseFileName = seriesDirectory + "/" + os.fsdecode(impulseFile)
            args = Arguments()
            args.fundumentalPitch = fundumentalPitch
            # librosa loading
            print("Loading file: " + impulseFileName)
            args.impulseLIB, samplingRate = librosa.load(impulseFileName, sr=48000)
            # Audio Samples are resampled to 48kHz to not cause problems for sub-band filtering and mfcc calculation
            
            #args.impulseLIB = librosa.to_mono(args.impulseLIB)
            # iracema loading
            args.impulseIRA = iracema.Audio(impulseFileName)
            args.impulseFFT = iracema.spectral.fft(args.impulseIRA, window_size=2048, hop_size=1024)
            args.pitch = iracema.pitch.hps(args.impulseFFT, minf0=50, maxf0=500)
            args.harmonicsIRA = iracema.harmonics.extract(args.impulseFFT, args.pitch)
            pitchesHz.append(np.median(args.pitch.data))

            if centroid_flag:
                centroids.append(np.mean(librosa.feature.spectral_centroid(args.impulseLIB)))
            if f0normCentroid_flag:
                if fundumentalPitch == 0:
                    f0normCentroids.append((np.mean(librosa.feature.spectral_centroid(args.impulseLIB) / np.median(args.pitch.data))))
                else:
                    f0normCentroids.append((np.mean(librosa.feature.spectral_centroid(args.impulseLIB) / fundumentalPitch)))
            if rolloff_flag:
                rolloffs.append(np.mean(librosa.feature.spectral_rolloff(args.impulseLIB)))
            if bandwidth_flag:
                bandwidths.append(np.mean(librosa.feature.spectral_bandwidth(args.impulseLIB)))
            if spread_flag:
                spreads.append(np.mean(iracema.features.spectral_spread(args.impulseFFT).data))
            if flux_flag:
                fluxes.append(np.mean(iracema.features.spectral_flux(args.impulseFFT).data))
            if subBandFlux_flag:
                subBandFlux = pc.CalculateSubBandSpectralFlux(args, samplingRate)
                subBandFluxes1.append(subBandFlux[0])
                subBandFluxes2.append(subBandFlux[1])
                subBandFluxes3.append(subBandFlux[2])
                subBandFluxes4.append(subBandFlux[3])
                subBandFluxes5.append(subBandFlux[4])
                subBandFluxes6.append(subBandFlux[5])
                subBandFluxes7.append(subBandFlux[6])
                subBandFluxes8.append(subBandFlux[7])
                if samplingRate > 22500:
                    subBandFluxes9.append(subBandFlux[8])
                if samplingRate > 45000:
                    subBandFluxes10.append(subBandFlux[9])
            if mfcc_flag:
                mfccs = librosa.feature.mfcc(args.impulseLIB, 48000, n_mfcc=13)
                # Teraz jeszcze tylko trzeba wiedzieć co z tym zrobić...
            if tuning_flag:
                tunings.append(np.mean(librosa.estimate_tuning(args.impulseLIB)))
            if crossingRate_flag:
                crossingRates.append(np.mean(librosa.feature.zero_crossing_rate(args.impulseLIB)))
            if rms_flag:
                rmss.append(pc.CalculateRMS(args))
            if entropy_flag:
                entropies.append(np.mean(iracema.features.spectral_entropy(args.impulseFFT).data))
            if temporalCentroid_flag:
                temporalCentroids.append(pc.CalculateTemporalCentroid(args))
            if logAttackTime_flag:
                logAttackTimes.append(pc.CalculateLogAttackTime(args))
            if decayTime_flag:
                decayTimes.append(pc.CalculateDecayTime(args))
            impulses = pc.InsertIntoVstack(args.impulseLIB, impulses)

        fullFrequencies, fullSpectrums, attackFrequencies, attackSpectrums, sustainFrequencies, sustainSpectrums, \
        decayFrequencies, decaySpectrums = pc.CalculateFFTs(impulses, samplingRate, attackTime, sustainTime)

        if fundumentalPitch == 0:
            fundumentalPitch = np.median(pitchesHz)
        foundFundumentalPitches.append(fundumentalPitch)
        mathHarmFreq = pc.CreateMathematicalHarmonicFrequencyVector(fundumentalPitch, n=20)
        harmonicData = pc.ExtractHarmonicDataFromSpectrums(fullSpectrums, fullFrequencies, mathHarmFreq, bufforInHZ=20)

        if noisiness_flag:
            noisinesses = pc.CalculateNoisiness(fullSpectrums, fullFrequencies, harmonicData)
        if highLowEnergy_flag:
            highLowEnergies = pc.CalculateHighEnergyLowEnergyRatio(fullSpectrums, fullFrequencies)
        if tristimulus_flag:
            tristimulus1s, tristimulus2s, tristimulus3s = pc.CalculateTristimulus(harmonicData)
        if inharmonicity_flag:
            inharmonicities = pc.CalculateInharmonicity(harmonicData)
        if oddeven_flag:
            oddEvenRatios = pc.CalculateOERs(harmonicData)

        # Dividing spectrum data into segments
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
        #maxAttack = max([maxAttack, max(avrAttackSpectrum)])
        #maxSustain = max([maxSustain, max(avrSustainSpectrum)])
        #maxDecay = max([maxDecay, max(avrDecaySpectrum)])

        seriesName = seriesDirectory.replace(inputDirectory + "/", "")
        series_names.append(seriesName)

        CalculateStatistics(centroids, centroid_values, centroid_deviations)
        CalculateStatistics(f0normCentroids, f0NormalizedCentroid_values, f0NormalizedCentroid_deviations)
        CalculateStatistics(rolloffs, rolloff_values, rolloff_deviations)
        CalculateStatistics(bandwidths, bandwidth_values, bandwidth_deviation)
        CalculateStatistics(spreads, spread_values, spread_deviations)
        CalculateStatistics(fluxes, flux_values, flux_deviations)
        CalculateStatistics(highLowEnergies, highLowEnergy_values, highLowEnegry_deviations)
        CalculateStatistics(subBandFluxes1, subBandFlux1_values, subBandFlux1_deviations)
        CalculateStatistics(subBandFluxes2, subBandFlux2_values, subBandFlux2_deviations)
        CalculateStatistics(subBandFluxes3, subBandFlux3_values, subBandFlux3_deviations)
        CalculateStatistics(subBandFluxes4, subBandFlux4_values, subBandFlux4_deviations)
        CalculateStatistics(subBandFluxes5, subBandFlux5_values, subBandFlux5_deviations)
        CalculateStatistics(subBandFluxes6, subBandFlux6_values, subBandFlux6_deviations)
        CalculateStatistics(subBandFluxes7, subBandFlux7_values, subBandFlux7_deviations)
        CalculateStatistics(subBandFluxes8, subBandFlux8_values, subBandFlux8_deviations)
        CalculateStatistics(subBandFluxes9, subBandFlux9_values, subBandFlux9_deviations)
        CalculateStatistics(subBandFluxes10, subBandFlux10_values, subBandFlux10_deviations)
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

    # -----------------Saving results-------------------
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
    if flux_flag:
        data_array = np.vstack((data_array, flux_values, flux_deviations))
    if highLowEnergy_flag:
        data_array = np.vstack((data_array, highLowEnergy_values, highLowEnegry_deviations))
    if subBandFlux_flag:
        data_array = np.vstack((data_array, subBandFlux1_values, subBandFlux1_deviations))
        data_array = np.vstack((data_array, subBandFlux2_values, subBandFlux2_deviations))
        data_array = np.vstack((data_array, subBandFlux3_values, subBandFlux3_deviations))
        data_array = np.vstack((data_array, subBandFlux4_values, subBandFlux4_deviations))
        data_array = np.vstack((data_array, subBandFlux5_values, subBandFlux5_deviations))
        data_array = np.vstack((data_array, subBandFlux6_values, subBandFlux6_deviations))
        data_array = np.vstack((data_array, subBandFlux7_values, subBandFlux7_deviations))
        data_array = np.vstack((data_array, subBandFlux8_values, subBandFlux8_deviations))
        if samplingRate > 22500:
            data_array = np.vstack((data_array, subBandFlux9_values, subBandFlux9_deviations))
        if samplingRate > 45000:
            data_array = np.vstack((data_array, subBandFlux10_values, subBandFlux10_deviations))
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

    # Saving data into .csv file
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
        if flux_flag:
            dataWriter.writerow(flux_values)
            dataWriter.writerow(flux_deviations)
        if highLowEnergy_flag:
            dataWriter.writerow(highLowEnergy_values)
            dataWriter.writerow(highLowEnegry_deviations)
        if subBandFlux_flag:
            dataWriter.writerow(subBandFlux1_values)
            dataWriter.writerow(subBandFlux1_deviations)
            dataWriter.writerow(subBandFlux2_values)
            dataWriter.writerow(subBandFlux2_deviations)
            dataWriter.writerow(subBandFlux3_values)
            dataWriter.writerow(subBandFlux3_deviations)
            dataWriter.writerow(subBandFlux4_values)
            dataWriter.writerow(subBandFlux4_deviations)
            dataWriter.writerow(subBandFlux5_values)
            dataWriter.writerow(subBandFlux5_deviations)
            dataWriter.writerow(subBandFlux6_values)
            dataWriter.writerow(subBandFlux6_deviations)
            dataWriter.writerow(subBandFlux7_values)
            dataWriter.writerow(subBandFlux7_deviations)
            dataWriter.writerow(subBandFlux8_values)
            dataWriter.writerow(subBandFlux8_deviations)
            if samplingRate > 22500:
                dataWriter.writerow(subBandFlux9_values)
                dataWriter.writerow(subBandFlux9_deviations)
            if samplingRate > 45000:
                dataWriter.writerow(subBandFlux10_values)
                dataWriter.writerow(subBandFlux10_deviations)
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

    spectrum_array, temp_array = [], []
    for iterator in range(0, len(seriesNames)):
        outputFile = seriesNames[iterator].replace(inputDirectory, outputDirectory)
        temp_array = [outputFile, allAttackSpectrums[iterator], \
                                    allAttackFrequencies[iterator], allSustainSpectrums[iterator], \
                                    allSustainFrequencies[iterator], allDecaySpectrums[iterator], \
                                    allDecayFrequencies[iterator]]

        spectrum_array.append(temp_array)


    np.save(outputDirectory + '/' + spectrumFileName + '_' + fileNameAppendix + '.npy', spectrum_array)









