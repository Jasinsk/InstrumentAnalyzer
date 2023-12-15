import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# This script draws the spectrums calculated and saved in ImpulseAnalyzer.py

def DrawSpectrum(frequencies, spectrum, maxValue, startTime, endTime, offset):
    plt.plot(frequencies + offset, spectrum, linewidth=1)
    #plt.locator_params(nbins=40)
    plt.xlim([0, 10000])
    #plt.ylim([10e-12, maxValue * 1.1])
    plt.xlabel('f [Hz]', fontsize=23)
    #plt.xticks(np.arange(110, 3110, step=100))
    ax = plt.gca()
    #ax.xaxis.set_major_locator(ticker.FixedLocator([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000]))
    #ax.xaxis.set_minor_locator(ticker.FixedLocator([110, 210, 310, 410, 510, 610, 710, 810, 910, 1010, 1110, 1210, 1310, 1410, 1510, 1610, 1710, 1810, 1910, 2010, 2110, 2210, 2310, 2410, 2510, 2610, 2710, 2810, 2910, 3010, 120, 220, 320, 420, 520, 620, 720, 820, 920, 1020, 1120, 1220, 1320, 1420, 1520, 1620, 1720, 1820, 1920, 2020, 2120, 2220, 2320, 2420, 2520, 2620, 2720, 2820, 2920, 3020, 130, 230, 330, 430, 530, 630, 730, 830, 930, 1030, 1130, 1230, 1330, 1430, 1530, 1630, 1730, 1830, 1930, 2030, 2130, 2230, 2330, 2430, 2530, 2630, 2730, 2830, 2930, 3030]))
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(which='major', axis='x', width=0.8, length=0, labelsize=8, labelrotation=90, grid_alpha=0.3)
    ax.tick_params(which='minor', width=0.8, length=0, color='0.5', grid_alpha=0.5)
    #plt.xticks(np.arange(100, 3100, step=100), fontsize=15, rotation=45)
    #plt.ticklabel_format(useMathText=False)
    #plt.xscale('log')
    #plt.ticklabel_format(useMathText=True, scilimits=(0, 0))
    plt.yticks(np.arange(-80, -35 , step=5),fontsize=14 )
    plt.ylim([-80, -40])
    plt.tick_params('x', length=0)
    #plt.title(str(startTime) + ' - ' + str(endTime) + ' [s]', fontsize=25)
    plt.grid(linewidth=1, which='both')
    plt.grid(True)
    #plt.yscale("log")
    # plt.text(1500, maxDecay, str('Energy = ' + str(CalculateRMS(allDecaySpectrums[iterator]))))

def run(inputDirectory, outputDirectory, fileNameAppendix, spectrumFileName, config, vectorOutput_flag):

    spectrumArray = np.load(f"{str(outputDirectory)}/{spectrumFileName}_{fileNameAppendix}.npy", allow_pickle=True)

    maxAttack, maxSustain, maxDecay, impulseTime = 0, 0, 0, 7.00

    scalingFactor = 1
    offset = 0
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

        plt.suptitle(seriesNames[series].replace(str(inputDirectory), ''), fontsize='xx-large')

        #converting frequencies to kHz for easier legibility
        kAttackFrequencies = allAttackFrequencies/1000
        kSustainFrequencies = allSustainFrequencies/1000
        kDecayFrequencies = allDecayFrequencies/1000
        kFullFrequencies = allFullFrequencies/1000

        ## Divided spectrums
        '''
        plt.subplot(131)
        plt.ylabel("Magnitude [dB FS]", fontsize = 32)
        #plt.ylabel("Amplituda [dB FS]", fontsize = 32)
        DrawSpectrum(scalingFactor * allAttackFrequencies, allAttackSpectrums, maxAttack, '0', config.attackCutTime, offset=0)
        plt.subplot(132)
        DrawSpectrum(scalingFactor * allSustainFrequencies, allSustainSpectrums, maxSustain, attackTime, config.sustainCutTime, offset=0)
        plt.subplot(133)
        DrawSpectrum(scalingFactor * allDecayFrequencies, allDecaySpectrums, maxDecay, sustainTime, round(impulseTime, 2), offset=0)
        '''
        # Single full spectrum
        plt.ylabel("Magnitude [dB FS]", fontsize=23)
        DrawSpectrum(scalingFactor * allFullFrequencies, allFullSpectrums, maxAttack, '0', round(impulseTime, 2), current_offset)
        current_offset += offset
        print("Outputing to: " + seriesNames)

        figure = plt.gcf()
        figure.set_size_inches(15, 8)

        #plt.legend(['No Modification', 'AVC in F14', 'AVC in K6', 'AVC in K6 with Phase Switch'], title='Configuration', title_fontsize=12, fontsize=12 )
        #plt.legend(['100 mm', '80 mm', '60 mm', '40 mm'], title='Sound Hole Diameter', title_fontsize=15, fontsize=15)
        #plt.legend(['0', '00', '1 D', '1 M', '1 O', '2 D', '2 M', '2 O', '3 D', '3 M', '3 O'], title='Configuration', title_fontsize=12, fontsize=10)

        plt.subplots_adjust(bottom=0.2)

        if vectorOutput_flag:
            plt.savefig(seriesNames + '.pdf', dpi=1200, format="pdf", bbox_inches='tight')
        else:
            plt.savefig(seriesNames, dpi = 100)


        #plt.show()
        plt.clf()


    plt.clf()