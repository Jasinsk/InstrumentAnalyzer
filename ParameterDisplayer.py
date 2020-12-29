import numpy as np
import matplotlib.pyplot as plt

# This script accepts a data file saved by ImpulseAnalyzer.py, displays the data and saves the plot into the output directory.
def run(inputDirectory, outputDirectory, fileNameAppendix, dataFileName):

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

                plt.errorbar(seriesNames, values, deviations, fmt='ko', ecolor='k', elinewidth=1.5, capsize=20)
                #plt.title(parameterName)
                plt.subplots_adjust(bottom=0.2)
                plt.grid(True)
                plt.yticks(fontsize=12)

                # Saving graph
                outputFile = outputDirectory + '/' + parameterName + '_' + fileNameAppendix
                figure = plt.gcf()
                figure.set_size_inches(7, 3.5)
                plt.savefig(outputFile, dpi=100)
                #plt.savefig(outputFile, dpi=100, format="eps")

                #plt.show()
                plt.clf()

                print('Figure saved as: ' + outputFile)