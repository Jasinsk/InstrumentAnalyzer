import numpy as np
import matplotlib.pyplot as plt

# This script accepts a data file saved by ImpulseAnalyzer.py, displays the data and saves the plot into the output directory.

def run(inputDirectory, outputDirectory, fileNameAppendix, dataFileName, vectorOutput_flag):

        customFontsize = 17

        dataArray = np.load(inputDirectory + '/' + dataFileName + '_' + fileNameAppendix + '.npy')
        seriesNames = dataArray[0,1:]
        dataArray = np.delete(dataArray, 0, 0)
        parameterNumber, sampleNumber = dataArray.shape

        for parameter in range((parameterNumber)//2):
                values = dataArray[parameter*2, 1:]
                values = values.astype(np.float)
                deviations = dataArray[(parameter*2)+1, 1:]
                deviations = deviations.astype(np.float)
                parameterName = dataArray[parameter*2,0]

                #Custom labels
                # Label for x axis
                plt.xlabel('Tonewood', fontsize=customFontsize)
                # Labels for x values
                plt.xticks([0, 1, 2, 3], ['Sapele', 'Pine', 'Plywood', 'Rosewood'], fontsize=15, rotation=0)
                #plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
                #           ['wn', 'wn-100', 'wn-300', 'wn-1500', 'wn-1500-2oct', 'wn-1500-3oct', 'wn-6000', 'sine-440', 'sine-447',
                #            'triangle-440', 'saw-440', 'sine-100,200,300,400', 'sine-100,210,320,390', 'a-2,d-8', 'a-0,5,d-9,5',
                #            'a-1,d-9', 'a-1,d-1', 'guitar'], fontsize=9, rotation=90)


                # assigning proper y axis labels to graphs
                if (parameterName == "Spectrum Centroid" or parameterName == "Bandwidth" or parameterName == "Rolloff"):
                        plt.ylabel('Frequency [Hz]', fontsize=customFontsize)
                elif (
                        parameterName == "RMS" or parameterName == "Attack RMS" or parameterName == "Sustain RMS" or parameterName == "Decay RMS"):
                        plt.ylabel('RMS', fontsize=customFontsize)
                elif (parameterName == "Tuning"):
                        plt.ylabel('Interval [cent]', fontsize=customFontsize)
                        values = values * 100
                        deviations = deviations * 100
                elif (parameterName == "Decay Time"):
                        plt.ylabel('Time [s]', fontsize=customFontsize)
                elif (parameterName == "Zero Crossing Rate"):
                        plt.ylabel('Rate', fontsize=customFontsize)

                plt.errorbar(seriesNames, values, deviations, fmt='ko', ecolor='k', elinewidth=1.5, capsize=20)
                plt.title(parameterName)
                plt.subplots_adjust(bottom=0.2)
                plt.grid(True)
                plt.yticks(fontsize=12)

                # Saving graph
                outputFile = outputDirectory + '/' + parameterName + '_' + fileNameAppendix
                figure = plt.gcf()
                figure.set_size_inches(7, 3.5)

                if vectorOutput_flag:
                        plt.savefig(outputFile, dpi=100, format="eps")
                else:
                        plt.savefig(outputFile, dpi=100)

                #plt.show()
                plt.clf()

                print('Figure saved as: ' + outputFile)