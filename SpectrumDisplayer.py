import numpy as np
import matplotlib.pyplot as plt

# This script draws the spectrums calculated and saved in ImpulseAnalyzer.py

def DrawSpectrum(frequencies, spectrum, maxValue, startTime, endTime):
    plt.plot(frequencies, spectrum, color='k')
    plt.locator_params(nbins=40)
    plt.xlim([0, 10000])
    #plt.ylim([10e-12, maxValue * 1.1])
    plt.xlabel('f [Hz]', fontsize=32)
    plt.xticks(fontsize=10)
    #plt.ticklabel_format(useMathText=True, scilimits=(0, 0))
    plt.yticks(fontsize=17)
    plt.ylim([-80, -20])
    #plt.title(str(startTime) + ' - ' + str(endTime) + ' [s]', fontsize=25)
    plt.grid(linewidth=1, which='both')
    #plt.yscale("log")
    # plt.text(1500, maxDecay, str('Energy = ' + str(CalculateRMS(allDecaySpectrums[iterator]))))

def run(inputDirectory, outputDirectory, fileNameAppendix, spectrumFileName, attackTime, sustainTime, vectorOutput_flag):

    spectrumArray = np.load(outputDirectory + '/' + spectrumFileName + '_' + fileNameAppendix + '.npy', allow_pickle=True)

    maxAttack, maxSustain, maxDecay, impulseTime = 0, 0, 0, 7.00

    scalingFactor = 1

    for series in range (0, len(spectrumArray)):

        seriesNames = spectrumArray[series][0]
        allAttackSpectrums = spectrumArray[series][1]
        allAttackFrequencies = spectrumArray[series][2]
        allSustainSpectrums = spectrumArray[series][3]
        allSustainFrequencies = spectrumArray[series][4]
        allDecaySpectrums = spectrumArray[series][5]
        allDecayFrequencies = spectrumArray[series][6]


        #plt.suptitle(seriesNames[iterator].replace(inputDirectory, ''), fontsize='xx-large')

        #converting frequencies to kHz for easier legibility
        kAttackFrequencies = allAttackFrequencies/1000
        kSustainFrequencies = allSustainFrequencies/1000
        kDecayFrequencies = allDecayFrequencies/1000

        #plt.subplot(131)
        #plt.ylabel("Magnitude [dB FS]", fontsize = 32)
        plt.ylabel("Amplituda [dB]", fontsize = 32)
        DrawSpectrum(scalingFactor * allAttackFrequencies, allAttackSpectrums, maxAttack, '0', attackTime)
        #plt.subplot(132)
        #DrawSpectrum(scalingFactor * allSustainFrequencies, allSustainSpectrums, maxSustain, attackTime, sustainTime)
        #plt.subplot(133)
        #DrawSpectrum(scalingFactor * allDecayFrequencies, allDecaySpectrums, maxDecay, sustainTime, round(impulseTime, 2))

        print("Outputing to: " + seriesNames)

        figure = plt.gcf()
        figure.set_size_inches(20, 8)

        if vectorOutput_flag:
            plt.savefig(seriesNames + '.pdf', dpi=1200, format="pdf")
        else:
            plt.savefig(seriesNames, dpi = 100)

        #plt.show()
        plt.clf()