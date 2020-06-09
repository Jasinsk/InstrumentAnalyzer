import numpy as np
import os
import matplotlib.pyplot as plt
import shutil

# This script accepts a data file saved by ImpulseAnalyzer.py, displays the data and saves the plot into the output directory.

inputDirectory = "AnalyzerOutputFolder"
outputDirectory = "DisplayerOutputFolder"
dataFileName = "ParameterData"

if os.path.isdir(outputDirectory):
        shutil.rmtree(outputDirectory)
os.mkdir(outputDirectory)

dataArray = np.load(inputDirectory + '/' + dataFileName + '.npy')
seriesNames = dataArray[0,1:]
dataArray = np.delete(dataArray, 0, 0)
parameterNumber, sampleNumber = dataArray.shape

for parameter in range((parameterNumber)//2):
        values = dataArray[parameter*2, 1:]
        values = values.astype(np.float)
        deviations = dataArray[(parameter*2)+1, 1:]
        deviations = deviations.astype(np.float)
        parameterName = dataArray[parameter*2,0]

        plt.errorbar(seriesNames, values, deviations, fmt='ko', ecolor='b', elinewidth=1.5, capsize=20)
        plt.title(parameterName)
        plt.grid(True)

        outputFile = outputDirectory + '/' + parameterName
        figure = plt.gcf()
        figure.set_size_inches(19.2, 10.8)
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
plt.grid(True)

outputFile = outputDirectory + '/' + 'Segment Energy Comparison'
figure = plt.gcf()
figure.set_size_inches(19.2, 10.8)
plt.savefig(outputFile, dpi=100)

#plt.show()