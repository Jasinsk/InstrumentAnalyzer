import ImpulseAnalyzer
import ParameterDisplayer

# -----------------ImpulseAnalyzer Controls-------------------
# Directories
analyzerInputDirectory = "ParserOutputFolder"
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

ImpulseAnalyzer.run(analyzerInputDirectory, analyzerOutputDirectory, dataFileName, attackCutTime, sustainCutTime, decayTime_flag)
ParameterDisplayer.run(displayerInputDirectory, displayerOutputDirectory, dataFileName)