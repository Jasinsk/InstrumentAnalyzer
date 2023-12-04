# InstrumentAnalyzer
Instrument Analyzer is a set of tools purpose build in Python to streamline the process of conducting comparative instrument analyses

## Setup

```Python == 3.9```

Instrument Analyzer requires the use of the Librosa package. At the time of writing there seems to be an issue with librosa installing on the build in virtual environment in PyCharm and VS19. 

If such problems arise it is advised to follow these steps:
1. install miniconda3: https://docs.conda.io/en/latest/miniconda.html
2. create a conda environment: conda create -n myenv
3. activate environment: source activate myenv
4. install librosa: conda install -c conda-forge librosa
5. set created environment as project interpreter

If you encounter audioread.NoBackendError please make sure that you have the ffmpeg library installed and it's location is available to Python in the environment variables. 

## Quick summary of use
Instrument Analyzer is divided into two modules. Signal parsing and signal analysis.

### Signal Parser

The parsing module was created to automate the proces of extracting impulses from a longer recording. It is simpler during recording to play many impulses of the same configuration one after another to guarantee a better statistical picture of the changes in question. This tools allows the user to load a single recording of multiple impulses into the input folder and it will parse each impulse from every recording and place them into folders with names that corespond to the names of the original audio files. The user has control over many parameters to make sure that the parser gets exactly what is expected of it. 
To use the parser simply place the recording files into the input folder, adjust the parameters so that the parser is accepting proper impulses and is cutting out what is needed in further analysis, adjust energy parameters in case there are any outlying recorded impulses and run SignalParser.py. The parsed impulses will show up in the designated output folder.

### Impulse Analysis

Group the parsed impulse folders in the way you want them compared on parameter graphs. Place each group in a seperate folder inside the designated input folder and simply run AnalysisController.py. The resulting graphs will show up in the analyzers output folder.

### Parameter References

Roughness - Daniel and Weber, 1997 

Loudness - DIN 45631/A1:2010; ISO 532-1:2017 §6
