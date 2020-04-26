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

