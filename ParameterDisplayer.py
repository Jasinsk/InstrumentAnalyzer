import numpy as np
import matplotlib.pyplot as plt

# This script accepts a data file saved by ImpulseAnalyzer.py, displays the data and saves the plot into the output directory.

def run(inputDirectory, outputDirectory, fileNameAppendix, dataFileName, vectorOutput_flag):

        customFontsize = 18

        dataArray = np.load(inputDirectory + '/' + dataFileName + '_' + fileNameAppendix + '.npy')
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

                #Custom labels
                # Label for x axis
                #plt.xlabel('Głębokość [mm]', fontsize=customFontsize)
                # Labels for x values
                #plt.xticks([0, 1, 2], ['30', '50', '65'])
                #plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], ['25', '30', '35', '40', '45', '50', '55', '60', '65', '70'])
                #plt.xticks([0, 1, 2, 3], ['Sapele', 'Pine', 'Plywood', 'Rosewood'], fontsize=15, rotation=0)
                plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
                           ['wn', 'wn-100', 'wn-300', 'wn-1500', 'wn-1500-2oct', 'wn-1500-3oct', 'wn-6000', 'sine-440', 'sine-447',
                            'triangle-440', 'saw-440', 'sine-100,200,300,400', 'sine-100,210,320,390', 'a-2,d-8', 'a-0,5,d-9,5',
                            'a-1,d-9', 'a-1,d-1', 'guitar'], fontsize=9, rotation=90)
                #plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17], ['klasyczna', 'elektryczna', 'fuzz', 'sitar',
                #                                'koto', 'klawesyn', 'pianino', 'wibrafon', 'marimba', 'skrzypce', 'kontrabas', 'fletnia pana',
                #                                'trabka', 'klarnet', 'organy', 'moog', 'FM', 'sinus + szum'], fontsize=8, rotation=90)
                #plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
                #           ['mahon nowy 7,2 7,6', 'mahon nowy 7,2 7,6 nie klejone siodelko', 'mahon 0 (stary) 7,4 7,5',
                #            'mahon 1 6,86 8,45', 'mahon 1 7.45 8.3', 'mahon 1 9,4 8.45', 'siwerk 0 7,3 7,9 oles robi zdj i video',
                #            'swierk 0 6,9 7,9', 'swierk 0 7,3 7,9', 'swierk 0 7,3 8,0', 'swierk 0 7,3 8,0 rozstroila sie',
                #            'swierk 1 7,35 7,2', 'swierk 2 7.2 7.6', 'swierk 2 7.2 7.6 drugi raz uciety poczotek'],
                #           fontsize=9, rotation=90)
                #plt.xticks([0, 1, 2, 3, 4, 5, 6, 7], ["pusta", "1 b", "1 s", "2 b", "2 s", "3 b", "3 s", "13"], fontsize=9, rotation=0)

                #plt.xticks([0, 1, 2, 3, 4, 5],
                #           ['wn', 'a-2,d-8', 'a-0,5,d-9,5', 'a-1,d-9', 'a-1,d-1', 'guitar'], fontsize=9, rotation=90)

                #plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                #           ['Alder 1', 'Alder 1.1', 'Alder 1.1 - repeat', 'Alder 2', 'Pine 1', 'Pine 2', 'Pine 3', 'Zebrano 1', 'Zebrano 1.1', 'Zebrano 2'], fontsize=9, rotation=90)

                #plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                #           ['0', '00', '1D', '1M', '1O','2D', '2M', '2O', '3D', '3M', '3O',], fontsize=9)

                #plt.xticks([0, 1, 2, 3], ['No Modification', 'F14', 'K6', 'K6 with Phase Shift'], fontsize=10, rotation=0)

                #plt.xticks([0,1,2,3], ["100", "80", "60", "40"], fontsize=12)

                #plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], ["Filc", "Guma", "Nylon 0.8 1", "Nylon 0.8 2", "Nylon 0.8 3", "Nylon 0.67", "Nylon 0.94", "Nylon 1.14", "Poliweglan", "Stal"], fontsize=9, rotation=45)

                #plt.xticks([0,1,2,3], ["094", "114", "067", "080"])

                # assigning proper y axis labels to graphs
                if (parameterName == "Spectrum Centroid" or parameterName == "Bandwidth" or parameterName == "Rolloff"):
                        plt.ylabel('Frequency [Hz]', fontsize=customFontsize)
                        #plt.ylabel('Częstotliwość [Hz]', fontsize=customFontsize)
                elif (parameterName == "Roughness"):
                        plt.ylabel('Roughness', fontsize=customFontsize)
                elif (parameterName == "Loudness Max"):
                        plt.ylabel('Maximum Loudness', fontsize=customFontsize)
                elif (parameterName == "Loudness Avr"):
                        plt.ylabel('Average Loudness', fontsize=customFontsize)
                elif (parameterName == "RMS" or parameterName == "Attack RMS" or parameterName == "Sustain RMS" or parameterName == "Decay RMS"):
                        plt.ylabel('RMS', fontsize=customFontsize)
                elif (parameterName == "Tuning"):
                        plt.ylabel('Interval [cent]', fontsize=customFontsize)
                        #plt.ylabel('Interwał [cent]', fontsize=customFontsize)
                        values = values * 100
                        deviations = deviations * 100
                elif (parameterName == "Decay Time"):
                        plt.ylabel('Time [s]', fontsize=customFontsize)
                        #plt.ylabel('Czas [s]', fontsize=customFontsize)
                elif (parameterName == "Zero Crossing Rate"):
                        plt.ylabel('Rate', fontsize=customFontsize)
                elif (parameterName == "Tristimulus 1" or parameterName == "Tristimulus 2" or parameterName == "Tristimulus 3"):
                        plt.ylabel('Tristimulus 3', fontsize=customFontsize)

                plt.errorbar(seriesNames, values, deviations, fmt='kD', ecolor='r', elinewidth=2, capthick=2, capsize=15, markersize=9)

                #plt.plot(seriesNames, values, 'ko')
                #plt.title(parameterName)
                plt.margins(0.03)
                plt.subplots_adjust(bottom=0.2)
                #plt.legend()
                plt.grid(True)
                plt.yticks(fontsize=15)
                plt.xticks(fontsize=17)
                plt.xlabel('Sound Hole Diameter [mm]', fontsize=18)

                # Saving graph
                outputFile = outputDirectory + '/' + parameterName + '_' + fileNameAppendix
                figure = plt.gcf()
                figure.set_size_inches(14, 10)

                if vectorOutput_flag:
                        plt.savefig(outputFile + '.pdf', dpi=1200, format="pdf")
                else:
                        plt.savefig(outputFile, dpi=100)

                #plt.show()
                plt.clf()

                print('Figure saved as: ' + outputFile)