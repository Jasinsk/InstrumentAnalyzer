import ImpulseAnalyzer
import ParameterDisplayer
import SpectrumDisplayer
from pathlib import Path
import os.path

# -----------------ImpulseAnalyzer Controls-------------------
# Directories
ANALYZER_INPUT_DIR = "AnalyzerInputFolder"
ANALYZER_OUTPUT_DIR = "AnalyzerOutputFolder"
PARAMETER_FILE_NAME = "ParameterData"
SPECTRUM_FILE_NAME = "SpectrumData"

class AnalyzerConfig:
    def __init__(self):
        # Flags used to decide which parameters will be calculated
        self.centroid_flag = True
        self.f0normCentroid_flag = True
        self.rolloff_flag = True
        self.bandwidth_flag = True
        self.spread_flag = True
        self.flux_flag = True
        self.irregularity_flag = True
        self.highLowEnergy_flag = True
        self.subBandFlux_flag = True
        self.tristimulus_flag = True
        self.inharmonicity_flag = True
        self.noisiness_flag = True
        self.oddeven_flag = True
        self.roughness_flag = True
        self.loudness_flag = True
        self.tuning_flag = True
        self.crossingRate_flag = True
        self.rms_flag = True
        self.entropy_flag = True
        self.temporalCentroid_flag = True
        self.logAttackTime_flag = True
        self.decayTime_flag = True
        self.mfcc_flag = True

        # The impulse is cut into three parts, beginning-attackTime, attackTime-sustainTime, sustainTime-end [s]
        self.attackCutTime = 1
        self.sustainCutTime = 3

# -----------------ParameterDisplayer Controls-------------------
# Directories
DISPLAYER_INPUT_DIR = ANALYZER_OUTPUT_DIR
DISPLAYER_OUTPUT_DIR = "DisplayerOutputFolder"

# -----------------Output Controls-------------------
# Decide whether the output graphs showed be in vector format
vectorOutput_flag = False

# Decide which sections should be run
analyses_flag = True
parameter_displayer_flag = True
spectrum_displayer_flag = True
# -----------------Running sections-------------------
analyzer_config = AnalyzerConfig()
for comparisonGroup in sorted(Path(ANALYZER_INPUT_DIR).iterdir()):

    if comparisonGroup.name != ".DS_Store": # ignore MacOS system files
        comparisonFolderName = Path(ANALYZER_INPUT_DIR) / comparisonGroup.name
        comparisonOutputDirectory = Path(ANALYZER_OUTPUT_DIR) / comparisonGroup.name

        if analyses_flag:
            ImpulseAnalyzer.run(comparisonFolderName, comparisonOutputDirectory, PARAMETER_FILE_NAME,
                                SPECTRUM_FILE_NAME, comparisonGroup.name, analyzer_config)

        if parameter_displayer_flag:
            ParameterDisplayer.run(comparisonOutputDirectory, comparisonOutputDirectory, comparisonGroup.name,
                                   PARAMETER_FILE_NAME, vectorOutput_flag)

        if spectrum_displayer_flag:
            SpectrumDisplayer.run(comparisonOutputDirectory, comparisonOutputDirectory, comparisonGroup.name,
                                   SPECTRUM_FILE_NAME, analyzer_config, vectorOutput_flag)