import ImpulseAnalyzer
import ParameterDisplayer
import SpectrumDisplayer
import os

# -----------------ImpulseAnalyzer Controls-------------------
# Directories
analyzerInputDirectory = "AnalyzerInputFolder"
analyzerOutputDirectory = "AnalyzerOutputFolder"
parameterFileName = "ParameterData"
spectrumFileName = "SpectrumData"

# The impulse is cut into three parts, beginning-attackTime, attackTime-sustainTime, sustainTime-end [s]
attackCutTime = 1
sustainCutTime = 4

# Flags used to decide which parameters will be calculated
centroid_flag = False
f0normCentroid_flag = False
rolloff_flag = False
bandwidth_flag = False
spread_flag = False
flux_flag = False
irregularity_flag = False
highLowEnergy_flag = False
subBandFlux_flag = False
tristimulus_flag = False
inharmonicity_flag = False
noisiness_flag = False
oddeven_flag = False
roughness_flag = True
loudness_flag = True
tuning_flag = False
crossingRate_flag = False
rms_flag = True
entropy_flag = False
temporalCentroid_flag = False
logAttackTime_flag = False
decayTime_flag= False
mfcc_flag = False

# -----------------ParameterDisplayer Controls-------------------
# Directories
displayerInputDirectory = analyzerOutputDirectory
displayerOutputDirectory = "DisplayerOutputFolder"

# -----------------Output Controls-------------------
# Decide whether the output graphs showed be in vector format
vectorOutput_flag = False

# Decide which sections should be run
analyses_flag = False
parameter_displayer_flag = True
spectrum_displayer_flag = False
# -----------------Running sections-------------------
for comparisonGroup in sorted(os.listdir(os.fsencode(analyzerInputDirectory))):

    if os.fsdecode(comparisonGroup) != ".DS_Store": # ignore MacOS system files
        comparisonFolderName = analyzerInputDirectory + "/" + os.fsdecode(comparisonGroup)
        comparisonOutputDirectory = analyzerOutputDirectory + "/" + os.fsdecode(comparisonGroup)

        if analyses_flag:
            ImpulseAnalyzer.run(comparisonFolderName, comparisonOutputDirectory, parameterFileName, spectrumFileName,
                os.fsdecode(comparisonGroup), attackCutTime, sustainCutTime, centroid_flag, f0normCentroid_flag, rolloff_flag,
                bandwidth_flag, spread_flag, flux_flag, irregularity_flag, highLowEnergy_flag, subBandFlux_flag, tristimulus_flag,
                inharmonicity_flag, noisiness_flag, oddeven_flag, roughness_flag, loudness_flag, tuning_flag, crossingRate_flag, rms_flag, entropy_flag,
                temporalCentroid_flag, logAttackTime_flag, decayTime_flag, mfcc_flag)

        if parameter_displayer_flag:
            ParameterDisplayer.run(comparisonOutputDirectory, comparisonOutputDirectory, os.fsdecode(comparisonGroup),
                                   parameterFileName, vectorOutput_flag)

        if spectrum_displayer_flag:
            SpectrumDisplayer.run(comparisonOutputDirectory, comparisonOutputDirectory, os.fsdecode(comparisonGroup),
                                   spectrumFileName, attackCutTime, sustainCutTime, vectorOutput_flag)