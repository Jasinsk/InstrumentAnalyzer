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

# -----------------ParameterDisplayer and SpectrumDisplayer Controls-------------------
# Directories
DISPLAYER_INPUT_DIR = ANALYZER_OUTPUT_DIR
DISPLAYER_OUTPUT_DIR = "DisplayerOutputFolder"


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


class DisplayerConfig:
    def __init__(self):
        # Decide whether the output graphs showed be in vector format
        self.vectorOutput_flag = False

        # Decide whether spectrums are divided between attack, sustain and decay phases
        self.dividedSpectrum_flag = False

        # Decide whiether frequencies on spectrums are in kHz
        self.spectrumInKilohertz = False


class ExecutionConfig:
    def __init__(self):
        # Decide whether signal analysis should run
        self.analyses_flag = True

        # Decide which display sections should be run
        self.parameter_displayer_flag = True
        self.spectrum_displayer_flag = True


def main():
    analyzer_config = AnalyzerConfig()
    displayer_config = DisplayerConfig()
    execution_config = ExecutionConfig()

    for comparison_path in sorted(Path(ANALYZER_INPUT_DIR).iterdir()):

        if comparison_path.name != ".DS_Store":  # ignore MacOS system files
            comparisonFolderName = Path(ANALYZER_INPUT_DIR) / comparison_path.name
            comparisonOutputDirectory = Path(ANALYZER_OUTPUT_DIR) / comparison_path.name

            if execution_config.analyses_flag:
                ImpulseAnalyzer.run(comparisonFolderName, comparisonOutputDirectory, PARAMETER_FILE_NAME,
                                    SPECTRUM_FILE_NAME, comparison_path.name, analyzer_config)

            if execution_config.parameter_displayer_flag:
                ParameterDisplayer.run(comparisonOutputDirectory, comparisonOutputDirectory, comparison_path.name,
                                       PARAMETER_FILE_NAME, displayer_config)

            if execution_config.spectrum_displayer_flag:
                SpectrumDisplayer.run(comparisonOutputDirectory, comparisonOutputDirectory, comparison_path.name,
                                      SPECTRUM_FILE_NAME, analyzer_config, displayer_config)


if __name__ == "__main__":
    main()
