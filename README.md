# InstrumentAnalyzer
Instrument Analyzer is a set of tools purpose build in Python to streamline the process of conducting comparative instrument analyses

## Setup
Instrument Analyzer requires the use of the Librosa package. At the time of writing their seems to be an issue with librosa instaling on the build in virtual environment in PyCharm and VS19. 

If such problems arise it is advised to follow these steps:
1. install miniconda3: https://docs.conda.io/en/latest/miniconda.html
2. create a conda environment: conda create -n myfirstenv
3. activate environment: source activate myfirstenv
4. install librosa: conda install -c conda-forge librosa
5. set created environment as project interpreter
