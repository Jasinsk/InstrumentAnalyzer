import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# This script draws the spectrums calculated and saved in ImpulseAnalyzer.py

def DrawSpectrum(frequencies, spectrum, maxValue, startTime, endTime, offset):
    plt.plot(frequencies + offset, spectrum, linewidth=1)
    plt.locator_params(nbins=40)
    plt.xlim([0, 3000])
    #plt.ylim([10e-12, maxValue * 1.1])
    plt.xlabel('f [Hz]', fontsize=23)
    #plt.xticks(np.arange(110, 3110, step=100))
    #ax = plt.gca()
    #ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
    plt.xticks(np.arange(100, 3100, step=100), fontsize=15, rotation=45)
    plt.ticklabel_format(useMathText=False)
    #plt.xscale('log')
    #plt.ticklabel_format(useMathText=True, scilimits=(0, 0))
    plt.yticks(np.arange(-80, -35 , step=5),fontsize=14 )
    plt.ylim([-80, -40])
    plt.tick_params('x', length=10)
    #plt.title(str(startTime) + ' - ' + str(endTime) + ' [s]', fontsize=25)
    plt.grid(linewidth=1, which='both')
    plt.grid(False)
    #plt.yscale("log")
    # plt.text(1500, maxDecay, str('Energy = ' + str(CalculateRMS(allDecaySpectrums[iterator]))))

def run(inputDirectory, outputDirectory, fileNameAppendix, spectrumFileName, attackTime, sustainTime, vectorOutput_flag):

    spectrumArray = np.load(outputDirectory + '/' + spectrumFileName + '_' + fileNameAppendix + '.npy', allow_pickle=True)

    maxAttack, maxSustain, maxDecay, impulseTime = 0, 0, 0, 7.00

    scalingFactor = 1
    offset = 10
    current_offset = 0

    for series in range (0, len(spectrumArray)):

        seriesNames = spectrumArray[series][0]
        allAttackSpectrums = spectrumArray[series][1]
        allAttackFrequencies = spectrumArray[series][2]
        allSustainSpectrums = spectrumArray[series][3]
        allSustainFrequencies = spectrumArray[series][4]
        allDecaySpectrums = spectrumArray[series][5]
        allDecayFrequencies = spectrumArray[series][6]
        allFullSpectrums = spectrumArray[series][7]
        allFullFrequencies = spectrumArray[series][8]

        #plt.suptitle(seriesNames[iterator].replace(inputDirectory, ''), fontsize='xx-large')

        #converting frequencies to kHz for easier legibility
        kAttackFrequencies = allAttackFrequencies/1000
        kSustainFrequencies = allSustainFrequencies/1000
        kDecayFrequencies = allDecayFrequencies/1000
        kFullFrequencies = allFullFrequencies/1000

        ## Divided spectrums
        # plt.subplot(131)
        # plt.ylabel("Magnitude [dB FS]", fontsize = 32)
        # #plt.ylabel("Amplituda [dB FS]", fontsize = 32)
        # DrawSpectrum(scalingFactor * allAttackFrequencies, allAttackSpectrums, maxAttack, '0', attackTime)
        # plt.subplot(132)
        # DrawSpectrum(scalingFactor * allSustainFrequencies, allSustainSpectrums, maxSustain, attackTime, sustainTime)
        # plt.subplot(133)
        # DrawSpectrum(scalingFactor * allDecayFrequencies, allDecaySpectrums, maxDecay, sustainTime, round(impulseTime, 2))

        # Single full spectrum
        plt.ylabel("Magnitude [dB FS]", fontsize=23)
        DrawSpectrum(scalingFactor * allFullFrequencies, allFullSpectrums, maxAttack, '0', round(impulseTime, 2), current_offset)
        current_offset += offset
        print("Outputing to: " + seriesNames)

        figure = plt.gcf()
        figure.set_size_inches(15, 8)

        #plt.legend(['No Modification', 'AVC in F14', 'AVC in K6', 'AVC in K6 with Phase Switch'], title='Configuration', title_fontsize=12, fontsize=12 )
        plt.legend(['100 mm', '80 mm', '60 mm', '40 mm'], title='Sound Hole Diameter', title_fontsize=15, fontsize=15)
        #plt.legend(['0', '00', '1 D', '1 M', '1 O', '2 D', '2 M', '2 O', '3 D', '3 M', '3 O'], title='Configuration', title_fontsize=12, fontsize=10)

        plt.subplots_adjust(bottom=0.2)

        if vectorOutput_flag:
            plt.savefig(seriesNames + '.pdf', dpi=1200, format="pdf", bbox_inches='tight')
        else:
            plt.savefig(seriesNames, dpi = 100)


        #plt.show()
        #plt.clf()


    plt.clf()