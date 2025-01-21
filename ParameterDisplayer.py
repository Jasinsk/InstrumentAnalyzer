"""
This script accepts a data file saved by ImpulseAnalyzer.py, displays the data and saves the plot into the output directory.
"""
import numpy as np
import matplotlib.pyplot as plt

def run(inputDirectory, outputDirectory, fileNameAppendix, dataFileName, displayer_config):

    customFontsize = 22
    xValueLabel = 'Material - Plucking Depth'

    dataArray = np.load(f"{str(inputDirectory)}/{dataFileName}_{fileNameAppendix}.npy")
    seriesNames = dataArray[0,1:]
    dataArray = np.delete(dataArray, 0, 0)
    parameterNumber, sampleNumber = dataArray.shape

# If not all parameters are drawn check here v
    for parameter in range(((parameterNumber-1)//2)+1):

        values = dataArray[parameter*2, 1:]
        values = values.astype(np.float)
        deviations = dataArray[(parameter*2)+1, 1:]
        deviations = deviations.astype(np.float)
        parameterName = dataArray[parameter*2,0]

        # # For MDPI article
        # values_temp = dataArray[parameter * 2, 1:]
        # values_temp = values_temp.astype(np.float)
        # deviations_temp = dataArray[(parameter * 2) + 1, 1:]
        # deviations_temp = deviations_temp.astype(np.float)
        # parameterName = dataArray[parameter * 2, 0]
        #
        # values = np.concatenate((
        #     values_temp[12:18],
        #     values_temp[24:30],
        #     values_temp[30:36],
        #     values_temp[42:48]
        # ))
        # deviations = np.concatenate((
        #     deviations_temp[12:18],
        #     deviations_temp[24:30],
        #     deviations_temp[30:36],
        #     deviations_temp[42:48]
        # ))
        # seriesNames = np.concatenate((
        #     seriesNames_temp[12:18],
        #     seriesNames_temp[24:30],
        #     seriesNames_temp[30:36],
        #     seriesNames_temp[42:48]
        # ))

        # Custom labels
        # Label for x axis
        #plt.xlabel('Głębokość [mm]', fontsize=customFontsize)

        # Ticks for x values
        """
        plt.xticks([0, 1, 2], ['30', '50', '65'])
        plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], ['25', '30', '35', '40', '45', '50', '55', '60', '65', '70'])
        plt.xticks([0, 1, 2, 3], ['Sapele', 'Pine', 'Plywood', 'Rosewood'], fontsize=15, rotation=0)
        
        plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
                   ['wn', 'wn-100', 'wn-300', 'wn-1500', 'wn-1500-2oct', 'wn-1500-3oct', 'wn-6000', 'sine-440', 'sine-447',
                    'triangle-440', 'saw-440', 'sine-100,200,300,400', 'sine-100,210,320,390', 'a-2,d-8', 'a-0,5,d-9,5',
                    'a-1,d-9', 'a-1,d-1', 'guitar'], fontsize=9, rotation=90)
        
        plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17], ['klasyczna', 'elektryczna', 'fuzz', 'sitar',
                                        'koto', 'klawesyn', 'pianino', 'wibrafon', 'marimba', 'skrzypce', 'kontrabas', 'fletnia pana',
                                        'trabka', 'klarnet', 'organy', 'moog', 'FM', 'sinus + szum'], fontsize=8, rotation=90)
    
            
        plt.xticks([0, 1, 2, 3, 4, 5, 6, 7], ["pusta", "1 b", "1 s", "2 b", "2 s", "3 b", "3 s", "13"], fontsize=9, rotation=0)
    
        plt.xticks([0, 1, 2, 3, 4, 5],
                   ['wn', 'a-2,d-8', 'a-0,5,d-9,5', 'a-1,d-9', 'a-1,d-1', 'guitar'], fontsize=9, rotation=90)
        
        plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                   ['Alder 1', 'Alder 1.1', 'Alder 1.1 - repeat', 'Alder 2', 'Pine 1', 'Pine 2', 'Pine 3', 'Zebrano 1', 'Zebrano 1.1', 'Zebrano 2'], fontsize=7, rotation=45)
        
        plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                   ['0', '00', '1D', '1M', '1O','2D', '2M', '2O', '3D', '3M', '3O',], fontsize=9)
    
        plt.xticks([0, 1, 2, 3], ['No Modification', 'F14', 'K6', 'K6 with Phase Shift'], fontsize=10, rotation=0)
            
        plt.xticks([0,1,2,3], ["100", "80", "60", "40"], fontsize=12)
        
        plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], ["Nylon 0.8", "Nylon 0.67", "Nylon 0.94", "Nylon 1.14", "Poliweglan", "Stal", "Guma", "Filc"], fontsize=9, rotation=45)
        
        plt.xticks([0,1,2,3], ["094", "114", "067", "080"])
        
        plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17], ["I-1", "I-2", "I-3", "I-4", "I-5", "I-6", "II-1", "II-2", "II-3", "II-4", "II-5", "II-6", "III-1", "III-2", "III-3", "III-4", "III-5", "III-6"], fontsize=9, rotation=0)

        plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                    18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
                    36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
                    54, 55, 56, 57, 58, 59],['I-1', 'I-2', 'I-3', 'I-4', 'I-5', 'I-6',
                    'II-1', 'II-2', 'II-3', 'II-4', 'II-5', 'II-6', 'III-1', 'III-2', 'III-3', 'III-4', 'III-5', 'III-6',
                    'IV-1', 'IV-2', 'IV-3', 'IV-4', 'IV-5', 'IV-6', 'V-1', 'V-2', 'V-3', 'V-4', 'V-5', 'V-6',
                    'VI-1', 'VI-2', 'VI-3', 'VI-4', 'VI-5', 'VI-6', 'VII-1', 'VII-2', 'VII-3', 'VII-4', 'VII-5', 'VII-6',
                    'VIII-1', 'VIII-2', 'VIII-3', 'VIII-4', 'VIII-5', 'VIII-6', 'IX-1', 'IX-2', 'IX-3', 'IX-4', 'IX-5', 'IX-6',
                    'X-1', 'X-2', 'X-3', 'X-4', 'X-5', 'X-6'])
        """
        plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                    18, 19, 20, 21, 22, 23], ['N-I', 'N-II', 'N-III', 'N-IV', 'N-V', 'N-VI',
                    'P-I', 'P-II', 'P-III', 'P-IV', 'P-V', 'P-VI',
                    'S-I', 'S-II', 'S-III', 'S-IV', 'S-V', 'S-VI',
                    'F-I', 'F-II', 'F-III', 'F-IV', 'F-V', 'F-VI'])
        """
        plt.xticks([0, 1, 2, 3, 4, 5], ["1", "2", "3", "4", "5", "6"])
        """

        # assigning proper y axis labels to graphs
        if (parameterName == "Spectrum Centroid" or parameterName == "Bandwidth" or parameterName == "Rolloff"):
            if displayer_config.graphsInPolish:
                    plt.ylabel('Częstotliwość [Hz]', fontsize=customFontsize)
            else:
                    plt.ylabel('Frequency [Hz]', fontsize=customFontsize)
        elif (parameterName == "Roughness"):
            if displayer_config.graphsInPolish:
                    plt.ylabel('Szorstkość', fontsize=customFontsize)
            else:
                    plt.ylabel('Roughness', fontsize=customFontsize)
        elif (parameterName == "Loudness Max"):
            if displayer_config.graphsInPolish:
                    plt.ylabel('Maksymalna Głośność', fontsize=customFontsize)
            else:
                    plt.ylabel('Maximum Loudness', fontsize=customFontsize)
        elif (parameterName == "Loudness Avr"):
            if displayer_config.graphsInPolish:
                    plt.ylabel('Średnia Głośność', fontsize=customFontsize)
            else:
                    plt.ylabel('Average Loudness', fontsize=customFontsize)
        elif (parameterName == "RMS" or parameterName == "Attack RMS" or parameterName == "Sustain RMS" or parameterName == "Decay RMS"):
            plt.ylabel('RMS', fontsize=customFontsize)
        elif (parameterName == "Tuning"):
            if displayer_config.graphsInPolish:
                plt.ylabel('Interwał [cent]', fontsize=customFontsize)
            else:
                plt.ylabel('Interval [cent]', fontsize=customFontsize)
            values = values * 100
            deviations = deviations * 100
        elif (parameterName == "Decay Time"):
            if displayer_config.graphsInPolish:
                plt.ylabel('Czas [s]', fontsize=customFontsize)
            else:
                plt.ylabel('Time [s]', fontsize=customFontsize)
        elif (parameterName == "Zero Crossing Rate"):
            plt.ylabel('Rate', fontsize=customFontsize)
        elif (parameterName == "Tristimulus 1" or parameterName == "Tristimulus 2" or parameterName == "Tristimulus 3"):
            plt.ylabel('Tristimulus', fontsize=customFontsize)

        # Graph settings
        plt.errorbar(seriesNames, values, deviations, fmt='kD', ecolor='r', elinewidth=0.1, capthick=1, capsize=15, markersize=15, alpha=1)

        #plt.title(parameterName)
        plt.margins(0.01)
        #plt.subplots_adjust(bottom=0.2)
        #plt.legend()
        plt.grid(True)
        plt.yticks(fontsize=20)
        plt.xticks(fontsize=16)
        plt.xlabel(xValueLabel, fontsize=18)

        # Special visuals
        # Make an error box connecting all error bars
        #plt.fill_between(seriesNames, values - deviations, values + deviations, alpha=0.1)
        # Highlight a section of the graph
        #plt.axvspan(xmin=10, xmax=15, facecolor='yellow', alpha=0.2)
        # Create colored boxes in place of error bars
        for i in range(len(seriesNames)):
            box_width = 0.01 * len(seriesNames)
            plt.fill_betweenx([values[i] - deviations[i], values[i] + deviations[i]], i - box_width, i + box_width, color='red', alpha=0.15)

        cmap = plt.get_cmap("tab10")
        colors = [cmap(i) for i in range(6)]

        # for i in range(0, 25, 6):
        #     x = range(24)
        #     plt.fill_between(x[i:i + 6], values[i:i + 6], color=colors[int(i/6)], alpha=0.3)

        # Saving graph
        outputFile = f"{str(outputDirectory)}/{parameterName}_{fileNameAppendix}"
        figure = plt.gcf()
        figure.set_size_inches(15, 9)

        if displayer_config.vectorOutput_flag:
            plt.savefig(outputFile + '.pdf', dpi=1200, format="pdf")
        else:
            plt.savefig(outputFile, dpi=100)

        #plt.show()
        plt.clf()

        print('Figure saved as: ' + outputFile)
