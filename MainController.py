import SignalParser
import ImpulseAnalyzer
import ParameterDisplayer

# -----------------SignalParser Controls-------------------
# Directories
parserInputDirectory = "InputFolder"
parserOutputDirectory = "ParserOutputFolder"

# Peak detection
threshold = 0.5
minimalTimeDifference = 1

# Impulse parsing
attackTime = 0.05
decayTime = 5

# Energy validation of impulses
acceptableEnergyDeviation = 0.3
attackEnergyTime = 1.5
attackEnergyDeviation = 0.25

# -----------------ImpulseAnalyzer Controls-------------------
# Directories
analyzerInputDirectory = parserOutputDirectory
analyzerOutputDirectory = "AnalyzerOutputFolder"
dataFileName = "ParameterData"

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

#SignalParser.run(inputDirectory, outputDirectory, threshold, minimalTimeDifference, attackTime, decayTime, acceptableEnergyDeviation, attackEnergyTime, attackEnergyDeviation)
ImpulseAnalyzer.run(analyzerInputDirectory, analyzerOutputDirectory, dataFileName, attackCutTime, sustainCutTime, decayTime_flag)
ParameterDisplayer.run(displayerInputDirectory, displayerOutputDirectory, dataFileName)