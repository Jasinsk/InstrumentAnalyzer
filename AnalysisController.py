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

# Flags used to decide which parameters will be calculated
centroid_flag = True
f0normCentroid_flag = True
rolloff_flag = True
bandwidth_flag = True
spread_flag = True
highLowEnergy_flag = True
tristimulus_flag = True
inharmonicity_flag = True
noisiness_flag = True
oddeven_flag = True
tuning_flag = True
crossingRate_flag = True
rms_flag = True
entropy_flag = True
temporalCentroid_flag = True
logAttackTime_flag = True
decayTime_flag= True

# -----------------ParameterDisplayer Controls-------------------
# Directories
displayerInputDirectory = analyzerOutputDirectory
displayerOutputDirectory = "DisplayerOutputFolder"

# -----------------Running sections-------------------
for comparisonGroup in os.listdir(os.fsencode(analyzerInputDirectory)):

    comparisonFolderName = analyzerInputDirectory + "/" + os.fsdecode(comparisonGroup)
    comparisonOutputDirectory = analyzerOutputDirectory + "/" + os.fsdecode(comparisonGroup)

    ImpulseAnalyzer.run(comparisonFolderName, comparisonOutputDirectory, parameterFileName, spectrumFileName,
        os.fsdecode(comparisonGroup), attackCutTime, sustainCutTime, centroid_flag, f0normCentroid_flag, rolloff_flag,
        bandwidth_flag, spread_flag, highLowEnergy_flag, tristimulus_flag, inharmonicity_flag, noisiness_flag,
        oddeven_flag, tuning_flag, crossingRate_flag, rms_flag, entropy_flag,
        temporalCentroid_flag, logAttackTime_flag, decayTime_flag)

    ParameterDisplayer.run(comparisonOutputDirectory, comparisonOutputDirectory, os.fsdecode(comparisonGroup), parameterFileName)