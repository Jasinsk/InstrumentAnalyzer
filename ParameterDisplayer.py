import numpy as np
import os
import matplotlib.pyplot as plt
import shutil

# This script accepts a data file saved by ImpulseAnalyzer.py, displays the data and saves the plot into the output directory.
def run(inputDirectory, outputDirectory, fileNameAppendix, dataFileName):

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

                if (parameterName == "Spectrum Centroid" or parameterName == "Bandwidth" or parameterName == "Rolloff"):
                        plt.ylabel('Frequency [Hz]', fontsize='x-large')
                elif (
                        parameterName == "RMS" or parameterName == "Attack RMS" or parameterName == "Sustain RMS" or parameterName == "Decay RMS"):
                        plt.ylabel('RMS', fontsize='x-large')
                elif (parameterName == "Tuning"):
                        plt.ylabel('Interval [cent]', fontsize='x-large')
                        values = values * 100
                        deviations = deviations * 100
                else:
                        plt.ylabel('Time [s]', fontsize='x-large')

                plt.errorbar(seriesNames, values, deviations, fmt='ko', ecolor='b', elinewidth=1.5, capsize=20)
                plt.title(parameterName)
                plt.xlabel('Tonewood', fontsize='x-large')
                plt.subplots_adjust(bottom=0.2)

                # Custom labels for x values
                plt.xticks([0, 1, 2, 3], ['Mahogony', 'Pine', 'Plywood', 'Rosewood'], fontsize='large', rotation=0)
                plt.grid(True)

                outputFile = outputDirectory + '/' + parameterName + '_' + fileNameAppendix
                figure = plt.gcf()
                figure.set_size_inches(10, 5.4)
                plt.savefig(outputFile, dpi=100)

                #plt.show()
                plt.clf()

                print('Figure saved as: ' + outputFile)


        # Displaying energy segment comparisons

        attackRMS = dataArray[parameterNumber - 6, 1:]
        attackRMS = attackRMS.astype(np.float)

        sustainRMS = dataArray[parameterNumber - 4, 1:]
        sustainRMS = sustainRMS.astype(np.float)

        decayRMS = dataArray[parameterNumber - 2, 1:]
        decayRMS = decayRMS.astype(np.float)

        # normalization of RMS values for easier comparison
        sustainRMS = sustainRMS * (np.mean(attackRMS)/np.mean(sustainRMS))
        decayRMS = decayRMS * (np.mean(attackRMS)/np.mean(decayRMS))

        plt.plot(seriesNames, attackRMS, 'ks')
        plt.plot(seriesNames, sustainRMS, 'bs')
        plt.plot(seriesNames, decayRMS, 'cs')
        plt.legend(['Attack', 'Sustain', 'Decay'])
        plt.title("Normalized Energy Segment Comparison")
        plt.yticks([])
        plt.grid(True)

        outputFile = outputDirectory + '/' + 'Segment Energy Comparison'
        figure = plt.gcf()
        figure.set_size_inches(19.2, 10.8)
        plt.savefig(outputFile, dpi=100)

        #plt.show()