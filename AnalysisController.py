import ImpulseAnalyzer
import ParameterDisplayer
import os

# -----------------ImpulseAnalyzer Controls-------------------
# Directories
analyzerInputDirectory = "AnalyzerInputFolder"
analyzerOutputDirectory = "AnalyzerOutputFolder"
parameterFileName = "ParameterData"
spectrumFileName = "SpectrumData"

# The impulse is cut into three parts, beginning-attackTime, attackTime-sustainTime, sustainTime-end
attackCutTime = 0.3
sustainCutTime = 1.3

# This variable decides whether the time intensive decay time calculations are conducted
decayTime_flag= False

# -----------------ParameterDisplayer Controls-------------------
# Directories
displayerInputDirectory = analyzerOutputDirectory
displayerOutputDirectory = "DisplayerOutputFolder"

# -----------------Running sections-------------------
for comparisonGroup in os.listdir(os.fsencode(analyzerInputDirectory)):

    comparisonFolderName = analyzerInputDirectory + "/" + os.fsdecode(comparisonGroup)
    comparisonOutputDirectory = analyzerOutputDirectory + "/" + os.fsdecode(comparisonGroup)

    ImpulseAnalyzer.run(comparisonFolderName, comparisonOutputDirectory, parameterFileName, spectrumFileName, os.fsdecode(comparisonGroup), attackCutTime, sustainCutTime, decayTime_flag)
    ParameterDisplayer.run(comparisonOutputDirectory, comparisonOutputDirectory, os.fsdecode(comparisonGroup), parameterFileName)