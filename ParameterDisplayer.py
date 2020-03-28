import numpy as np
from scipy import signal
import os
import matplotlib.pyplot as plt
import librosa
import librosa.display
import math
import shutil
import csv



inputDirectory = "AnalyzerOutputFolder"
outputDirectory = "DisplayerOutputFolder"
dataFileName = "ParameterData"

if os.path.isdir(outputDirectory):
        shutil.rmtree(outputDirectory)
os.mkdir(outputDirectory)

dataArray = np.load(inputDirectory + '/' + dataFileName + '.npy')
seriesNames = dataArray[0,1:]
print(seriesNames)
dataArray = np.delete(dataArray, 0, 0)
parameterNumber, sampleNumber = dataArray.shape

for parameter in range((parameterNumber)//2):
        values = dataArray[parameter*2, 1:]
        values = values.astype(np.float)
        deviations = dataArray[(parameter*2)+1, 1:]
        deviations = deviations.astype(np.float)
        parameterName = dataArray[parameter*2,0]

        plt.errorbar(seriesNames, values, deviations)
        plt.show()

