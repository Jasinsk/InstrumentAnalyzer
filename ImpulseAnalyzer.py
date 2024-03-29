"""
This script accepts folders with individually parsed impulses and calculates averaged spectra as well as
spectral and energy parameters. The results of these calculations are saved in the output directory for further use
with displayers
"""
import numpy as np
import librosa
import librosa.display
import iracema
import shutil
import csv
import ParameterCalculator as pc
from pathlib import Path


class Arguments:
    pass


class ParameterValues:
    def __init__(self, name, flag):
        self.values = [str(name)]
        self.deviations = [f"{str(name)} deviations"]
        self.flag = flag


class ParameterData:
    def __init__(self, config):
        self.ParameterValuesDict = {
            "centroid":         ParameterValues("Spectrum Centroid", config.centroid_flag),
            "f0NormalizedCentroid": ParameterValues("F0 Normalized Spectral Centroid", config.f0NormCentroid_flag),
            "rolloff":          ParameterValues("Rolloff", config.rolloff_flag),
            "bandwidth":        ParameterValues("Bandwidth", config.bandwidth_flag),
            "spread":           ParameterValues("Spread", config.spread_flag),
            "flux":             ParameterValues("Spectral Flux", config.flux_flag),
            "irregularity":     ParameterValues("Spectral Irregularity", config.irregularity_flag),
            "highLowEnergy":    ParameterValues("High Energy - Low Energy Ratio", config.highLowEnergy_flag),
            "subBandFlux1":     ParameterValues("Sub-Band Flux 1", config.subBandFlux_flag),
            "subBandFlux2":     ParameterValues("Sub-Band Flux 2", config.subBandFlux_flag),
            "subBandFlux3":     ParameterValues("Sub-Band Flux 3", config.subBandFlux_flag),
            "subBandFlux4":     ParameterValues("Sub-Band Flux 4", config.subBandFlux_flag),
            "subBandFlux5":     ParameterValues("Sub-Band Flux 5", config.subBandFlux_flag),
            "subBandFlux6":     ParameterValues("Sub-Band Flux 6", config.subBandFlux_flag),
            "subBandFlux7":     ParameterValues("Sub-Band Flux 7", config.subBandFlux_flag),
            "subBandFlux8":     ParameterValues("Sub-Band Flux 8", config.subBandFlux_flag),
            "subBandFlux9":     ParameterValues("Sub-Band Flux 9", config.subBandFlux_flag),
            "subBandFlux10":    ParameterValues("Sub-Band Flux 10", config.subBandFlux_flag),
            "tristimulus1":     ParameterValues("Tristimulus 1", config.tristimulus_flag),
            "tristimulus2":     ParameterValues("Tristimulus 2", config.tristimulus_flag),
            "tristimulus3":     ParameterValues("Tristimulus 3", config.tristimulus_flag),
            "inharmonicity":    ParameterValues("Inharmonicity", config.inharmonicity_flag),
            "noisiness":        ParameterValues("Noisiness", config.noisiness_flag),
            "oddEvenRatio":     ParameterValues("Odd-Even Ratio", config.oddEven_flag),
            "roughness":        ParameterValues("Roughness", config.roughness_flag),
            "loudnessMax":      ParameterValues("Loudness Max", config.loudness_flag),
            "loudnessAvr":      ParameterValues("Loudness Avr", config.loudness_flag),
            "tuning":           ParameterValues("Tuning", config.tuning_flag),
            "zeroCrossingRate": ParameterValues("Zero Crossing Rate", config.zeroCrossingRate_flag),
            "rms":              ParameterValues("RMS", config.rms_flag),
            "entropy":          ParameterValues("Entropy", config.entropy_flag),
            "temporalCentroid": ParameterValues("Temporal Centroid", config.temporalCentroid_flag),
            "logAttackTime":    ParameterValues("Log Attack Time", config.logAttackTime_flag),
            "decayTime":        ParameterValues("Decay Time", config.decayTime_flag),
            "mfcc1_means":      ParameterValues("MFCC 1 - mean", config.mfcc_flag),
            "mfcc1_stddevs":    ParameterValues("MFCC 1 - STDDEV", config.mfcc_flag),
            "mfcc2_means":      ParameterValues("MFCC 2 - mean", config.mfcc_flag),
            "mfcc2_stddevs":    ParameterValues("MFCC 2 - STDDEV", config.mfcc_flag),
            "mfcc3_means":      ParameterValues("MFCC 3 - mean", config.mfcc_flag),
            "mfcc3_stddevs":    ParameterValues("MFCC 3 - STDDEV", config.mfcc_flag),
            "mfcc4_means":      ParameterValues("MFCC 4 - mean", config.mfcc_flag),
            "mfcc4_stddevs":    ParameterValues("MFCC 4 - STDDEV", config.mfcc_flag),
        }

    def AppendSeriesStatistics(self, seriesData):
        for key, data in self.ParameterValuesDict.items():
            data.values.append(np.mean(seriesData.SeriesDict[key]))
            data.deviations.append(np.std(seriesData.SeriesDict[key]))

    def SaveData(self, data_array, outputDirectory, parameterFileName, fileNameAppendix):
        for key, data in self.ParameterValuesDict.items():
            if data.flag:
                data_array = np.vstack((data_array, data.values, data.deviations))

        np.save(f"{str(outputDirectory)}/{parameterFileName}_{fileNameAppendix}.npy", data_array)

    def SaveCSV(self, series_names, outputDirectory, parameterFileName, fileNameAppendix, foundFundumentalPitches):
        with open(f"{str(outputDirectory)}/{parameterFileName}_{fileNameAppendix}.csv", 'w', newline='') as csvfile:
            dataWriter = csv.writer(csvfile, delimiter=',', quotechar=';', quoting=csv.QUOTE_MINIMAL)
            dataWriter.writerow(series_names)
            for key, data in self.ParameterValuesDict.items():
                if data.flag:
                    dataWriter.writerow(data.values)
                    dataWriter.writerow(data.deviations)
            dataWriter.writerow(foundFundumentalPitches)

        print(f"Data saved to: {parameterFileName}_{fileNameAppendix}")


class SeriesData:
    def __init__(self):
        self.SeriesDict = {
            "centroid": [],
            "f0NormalizedCentroid": [],
            "rolloff": [],
            "bandwidth": [],
            "spread": [],
            "flux": [],
            "irregularity": [],
            "highLowEnergy": [],
            "subBandFlux1": [],
            "subBandFlux2": [],
            "subBandFlux3": [],
            "subBandFlux4": [],
            "subBandFlux5": [],
            "subBandFlux6": [],
            "subBandFlux7": [],
            "subBandFlux8": [],
            "subBandFlux9": [],
            "subBandFlux10": [],
            "tristimulus1": [],
            "tristimulus2": [],
            "tristimulus3": [],
            "inharmonicity": [],
            "noisiness": [],
            "oddEvenRatio": [],
            "roughness": [],
            "loudnessMax": [],
            "loudnessAvr": [],
            "tuning": [],
            "zeroCrossingRate": [],
            "rms": [],
            "entropy": [],
            "temporalCentroid": [],
            "logAttackTime": [],
            "decayTime": [],
            "mfcc1_means": [],
            "mfcc1_stddevs": [],
            "mfcc2_means": [],
            "mfcc2_stddevs": [],
            "mfcc3_means": [],
            "mfcc3_stddevs": [],
            "mfcc4_means": [],
            "mfcc4_stddevs": [],
        }


class Spectrum:
    def __init__(self):
        self.values = []
        self.frequencies = []


class SpectrumData:
    def __init__(self):
        self.SpectrumDict = {
            "AttackSpectrums": Spectrum(),
            "SustainSpectrums": Spectrum(),
            "DecaySpectrums": Spectrum(),
            "FullSpectrums": Spectrum()
        }

    def SaveData(self, seriesNames, inputDirectory, outputDirectory, spectrumFileName, fileNameAppendix):
        spectrum_array = []
        for i in range(0, len(seriesNames)):
            outputFile = seriesNames[i].replace(str(inputDirectory), str(outputDirectory))
            spectrum_array.append([outputFile,
                          self.SpectrumDict["AttackSpectrums"].values[i],
                          self.SpectrumDict["AttackSpectrums"].frequencies[i],
                          self.SpectrumDict["SustainSpectrums"].values[i],
                          self.SpectrumDict["SustainSpectrums"].frequencies[i],
                          self.SpectrumDict["DecaySpectrums"].values[i],
                          self.SpectrumDict["DecaySpectrums"].frequencies[i],
                          self.SpectrumDict["FullSpectrums"].values[i],
                          self.SpectrumDict["FullSpectrums"].frequencies[i]
                          ])

        np.save(f"{str(outputDirectory)}/{spectrumFileName}_{fileNameAppendix}.npy", spectrum_array)

    def SaveCSV(self, seriesNames, outputDirectory, spectrumFileName, fileNameAppendix):
        with open(f"{str(outputDirectory)}/{spectrumFileName}_{fileNameAppendix}.csv", 'w', newline='') as csvfile:
            dataWriter = csv.writer(csvfile, delimiter=',', quotechar=';', quoting=csv.QUOTE_MINIMAL)
            for i in range(0, len(seriesNames)):
                dataWriter.writerow("Name: ")
                dataWriter.writerow(seriesNames[i])
                dataWriter.writerow("Attack spectrum: ")
                dataWriter.writerow(self.SpectrumDict["AttackSpectrums"].values[i])
                dataWriter.writerow("Attack frequencies: ")
                dataWriter.writerow(self.SpectrumDict["AttackSpectrums"].frequencies[i])
                dataWriter.writerow("Sustain spectrum: ")
                dataWriter.writerow(self.SpectrumDict["SustainSpectrums"].values[i])
                dataWriter.writerow("Sustain frequencies: ")
                dataWriter.writerow(self.SpectrumDict["SustainSpectrums"].frequencies[i])
                dataWriter.writerow("Decay Spectrum: ")
                dataWriter.writerow(self.SpectrumDict["DecaySpectrums"].values[i])
                dataWriter.writerow("Decay frequencies: ")
                dataWriter.writerow(self.SpectrumDict["DecaySpectrums"].frequencies[i])
                dataWriter.writerow("Full Spectrum: ")
                dataWriter.writerow(self.SpectrumDict["FullSpectrums"].values[i])
                dataWriter.writerow("Full frequencies: ")
                dataWriter.writerow(self.SpectrumDict["FullSpectrums"].frequencies[i])

        print(f"Spectrums saved to: {spectrumFileName}_{fileNameAppendix}")


def run(inputDirectory, outputDirectory, parameterFileName, spectrumFileName, fileNameAppendix, config):

    # Clear output folder
    if Path(outputDirectory).is_dir():
        shutil.rmtree(outputDirectory)
    print(outputDirectory)
    Path(outputDirectory).mkdir()

    # ----------Setting up variables required for calculation and saving of parameter data---------------
    series_names = [" "]
    foundFundamentalPitches = ["Average Found Fundamental Pitches"]
    parameterData = ParameterData(config)
    spectrumData = SpectrumData()
    seriesNames = []

    # ---------------Calculating spectrums and parameters------------------
    for seriesDirectory in sorted(Path(inputDirectory).iterdir()):
        if seriesDirectory.name != ".DS_Store": # ignore MacOS system files
            seriesDirectory = Path(inputDirectory) / seriesDirectory.name
            print("Entering folder: " + seriesDirectory.name)

            seriesData = SeriesData()
            impulses, attackSpectrums, sustainSpectrums, decaySpectrums, pitchesHz = [], [], [], [], []

            for impulseFile in sorted(Path(seriesDirectory).iterdir()):
                impulseFileName = Path(seriesDirectory) / impulseFile.name
                args = Arguments()
                args.fundamentalFrequency = config.fundamentalFrequency
                # librosa loading
                print("Loading file: " + str(impulseFileName))
                args.impulseLIB, args.samplingRate = librosa.load(impulseFileName, sr=48000)
                # Audio Samples are resampled to 48kHz to not cause problems for sub-band filtering and mfcc calculation

                # iracema loading
                args.impulseIRA = iracema.Audio(str(impulseFileName))
                args.impulseFFT = iracema.spectral.fft(args.impulseIRA, window_size=2048, hop_size=1024)
                args.pitch = iracema.pitch.expan_pitch(args.impulseFFT, minf0=50, maxf0=500)
                args.harmonicsIRA = iracema.harmonics.extract(args.impulseFFT, args.pitch)
                pitchesHz.append(np.median(args.pitch.data))

                if config.centroid_flag:
                    seriesData.SeriesDict["centroid"].append(np.mean(librosa.feature.spectral_centroid(args.impulseLIB, sr=args.samplingRate)))
                if config.f0NormCentroid_flag:
                    if not config.fundamentalFrequency:
                        seriesData.SeriesDict["f0NormalizedCentroid"].append((np.mean(librosa.feature.spectral_centroid(args.impulseLIB, sr=args.samplingRate) / np.median(args.pitch.data))))
                    else:
                        seriesData.SeriesDict["f0NormalizedCentroid"].append((np.mean(librosa.feature.spectral_centroid(args.impulseLIB, sr=args.samplingRate) / config.fundamentalFrequency)))
                if config.rolloff_flag:
                    seriesData.SeriesDict["rolloff"].append(np.mean(librosa.feature.spectral_rolloff(args.impulseLIB, sr=args.samplingRate)))
                if config.bandwidth_flag:
                    seriesData.SeriesDict["bandwidth"].append(np.mean(librosa.feature.spectral_bandwidth(args.impulseLIB, sr=args.samplingRate)))
                if config.spread_flag:
                    seriesData.SeriesDict["spread"].append(np.mean(iracema.features.spectral_spread(args.impulseFFT).data))
                if config.flux_flag:
                    seriesData.SeriesDict["flux"].append(np.mean(iracema.features.spectral_flux(args.impulseFFT).data).real)
                if config.subBandFlux_flag:
                    subBandFlux = pc.CalculateSubBandSpectralFlux(args, args.samplingRate)
                    seriesData.SeriesDict["subBandFlux1"].append(subBandFlux[0])
                    seriesData.SeriesDict["subBandFlux2"].append(subBandFlux[1])
                    seriesData.SeriesDict["subBandFlux3"].append(subBandFlux[2])
                    seriesData.SeriesDict["subBandFlux4"].append(subBandFlux[3])
                    seriesData.SeriesDict["subBandFlux5"].append(subBandFlux[4])
                    seriesData.SeriesDict["subBandFlux6"].append(subBandFlux[5])
                    seriesData.SeriesDict["subBandFlux7"].append(subBandFlux[6])
                    seriesData.SeriesDict["subBandFlux8"].append(subBandFlux[7])
                    if args.samplingRate > 22500:
                        seriesData.SeriesDict["subBandFlux9"].append(subBandFlux[8])
                    if args.samplingRate > 45000:
                        seriesData.SeriesDict["subBandFlux10"].append(subBandFlux[9])
                if config.roughness_flag:
                    seriesData.SeriesDict["roughness"].append(pc.CalculateRoughness(args))
                if config.loudness_flag:
                    loudnessMax, loudnessAvr = pc.CalculateLoudness(args)
                    seriesData.SeriesDict["loudnessMax"].append(loudnessMax)
                    seriesData.SeriesDict["loudnessAvr"].append(loudnessAvr)
                if config.tuning_flag:
                    seriesData.SeriesDict["tuning"].append(np.mean(librosa.estimate_tuning(args.impulseLIB, sr=args.samplingRate)))
                if config.zeroCrossingRate_flag:
                    seriesData.SeriesDict["zeroCrossingRate"].append(np.mean(librosa.feature.zero_crossing_rate(args.impulseLIB)))
                if config.rms_flag:
                    seriesData.SeriesDict["rms"].append(pc.CalculateRMS(args))
                if config.entropy_flag:
                    seriesData.SeriesDict["entropy"].append(np.mean(iracema.features.spectral_entropy(args.impulseFFT).data))
                if config.temporalCentroid_flag:
                    seriesData.SeriesDict["temporalCentroid"].append(pc.CalculateTemporalCentroid(args))
                if config.logAttackTime_flag:
                    seriesData.SeriesDict["logAttackTime"].append(pc.CalculateLogAttackTime(args))
                if config.decayTime_flag:
                    seriesData.SeriesDict["decayTime"].append(pc.CalculateDecayTime(args))
                if config.mfcc_flag:
                    mfccs = librosa.feature.mfcc(args.impulseLIB, sr=args.samplingRate, n_mfcc=4)
                    seriesData.SeriesDict["mfcc1_means"].append(np.mean(mfccs[0]))
                    seriesData.SeriesDict["mfcc1_stddevs"].append(np.std(mfccs[0]))
                    seriesData.SeriesDict["mfcc2_means"].append(np.mean(mfccs[1]))
                    seriesData.SeriesDict["mfcc2_stddevs"].append(np.std(mfccs[1]))
                    seriesData.SeriesDict["mfcc3_means"].append(np.mean(mfccs[2]))
                    seriesData.SeriesDict["mfcc3_stddevs"].append(np.std(mfccs[2]))
                    seriesData.SeriesDict["mfcc4_means"].append(np.mean(mfccs[3]))
                    seriesData.SeriesDict["mfcc4_stddevs"].append(np.std(mfccs[3]))
                impulses = pc.InsertIntoVstack(args.impulseLIB, impulses)

            fullFrequencies, fullSpectrums, attackFrequencies, attackSpectrums, sustainFrequencies, sustainSpectrums, \
            decayFrequencies, decaySpectrums = pc.CalculateFFTs(impulses, args.samplingRate, config.attackCutTime, config.sustainCutTime)

            if not config.fundamentalFrequency:
                fundamentalFrequency = np.median(pitchesHz)
                print(f"Found fundumental frequency: {fundamentalFrequency}")
            else:
                fundamentalFrequency = config.fundamentalFrequency

            foundFundamentalPitches.append(fundamentalFrequency)
            mathHarmFreq = pc.CreateMathematicalHarmonicFrequencyVector(fundamentalFrequency, n=20)
            harmonicData = pc.ExtractHarmonicDataFromSpectrums(fullSpectrums, fullFrequencies, mathHarmFreq, bufforInHZ=20)

            if config.noisiness_flag:
                seriesData.SeriesDict["noisiness"] = pc.CalculateNoisiness(fullSpectrums, fullFrequencies, harmonicData)
            if config.highLowEnergy_flag:
                seriesData.SeriesDict["highLowEnergy"] = pc.CalculateHighEnergyLowEnergyRatio(fullSpectrums, fullFrequencies)
            if config.irregularity_flag:
                seriesData.SeriesDict["irregularity"] = pc.CalculateIrregularity(harmonicData)
            if config.tristimulus_flag:
                seriesData.SeriesDict["tristimulus1"], seriesData.SeriesDict["tristimulus2"], \
                seriesData.SeriesDict["tristimulus3"] = pc.CalculateTristimulus(harmonicData)
            if config.inharmonicity_flag:
                seriesData.SeriesDict["inharmonicity"] = pc.CalculateInharmonicity(harmonicData)
            if config.oddEven_flag:
                seriesData.SeriesDict["oddEvenRatio"] = pc.CalculateOERs(harmonicData)

            # Dividing spectrum data into segments
            avrAttackSpectrum = pc.CalculateAverageVector(attackSpectrums)
            avrSustainSpectrum = pc.CalculateAverageVector(sustainSpectrums)
            avrDecaySpectrum = pc.CalculateAverageVector(decaySpectrums)
            avrFullSpectrum = pc.CalculateAverageVector(fullSpectrums)

            spectrumData.SpectrumDict["AttackSpectrums"].values.append(avrAttackSpectrum)
            spectrumData.SpectrumDict["SustainSpectrums"].values.append(avrSustainSpectrum)
            spectrumData.SpectrumDict["DecaySpectrums"].values.append(avrDecaySpectrum)
            spectrumData.SpectrumDict["FullSpectrums"].values.append(avrFullSpectrum)
            spectrumData.SpectrumDict["AttackSpectrums"].frequencies.append(attackFrequencies)
            spectrumData.SpectrumDict["SustainSpectrums"].frequencies.append(sustainFrequencies)
            spectrumData.SpectrumDict["DecaySpectrums"].frequencies.append(decayFrequencies)
            spectrumData.SpectrumDict["FullSpectrums"].frequencies.append(fullFrequencies)
            seriesNames.append(str(seriesDirectory))

            series_names.append(seriesDirectory.name)

            parameterData.AppendSeriesStatistics(seriesData)

    # -----------------Saving results-------------------
    # Saving parameter data into .npy file
    data_array = series_names
    parameterData.SaveData(data_array, outputDirectory, parameterFileName, fileNameAppendix)

    # Saving data into .csv file
    parameterData.SaveCSV(data_array, outputDirectory, parameterFileName, fileNameAppendix, foundFundamentalPitches)

    # Saving spectrum data
    spectrumData.SaveData(seriesNames, inputDirectory, outputDirectory, spectrumFileName, fileNameAppendix)

    # Saving spectrum data into .csv file
    spectrumData.SaveCSV(seriesNames, outputDirectory, spectrumFileName, fileNameAppendix)


