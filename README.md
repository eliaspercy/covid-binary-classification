
Python code comprising my implementation of three machine learning models, utilising the dataset from https://github.com/beoutbreakprepared/nCoV2019/tree/master/latest_data, that serve the predict the relative outcome of coronavirus patients given background information. Also includes a report detailing the process of data analysis and cleaning, as well as the decisions underlying the machine learning models themselves.


RUNNING THE PROGRAM
-------------------

The codebase has been written entirely in Python 3.9, on a Windows 10 OS.

The following is a list of external libraries used in the codebase:

    imbalanced_learn==0.8.0
    pandas==1.2.2
    matplotlib==3.3.4
    numpy==1.20.1
    seaborn==0.11.1
    imblearn==0.0
    scikit_learn==0.24.2

These can each be installed in the usual way via pip, e.g.:

    pip install imbalanced_learn==0.8.0

Please ensure the version of the library is specified, like in the above
example, as these are the precise versions used to write the code.

Once these libraries have been installed, the program can be run in the usual
way from the terminal, i.e.:

    python covid_ml.py



THE DATA (latestdata.csv)
-------------------------

The program does not require the dataset to be pre-installed, instead, it does
require an internet connection because the dataset will automatically be
downloaded, and placed in a local folder entitled 'data'.
If internet access is unavailable but the dataset is installed, please place
the dataset in a folder in the same directory as the Python file entitled
'data', and change the DOWNLOAD_DATASET parameter below to False.



Additional notes
----------------

In terms of runtime, the program will generally take up to 15 minutes to
complete, which entails the training of all three models and the plotting of
various visualisations for exploration and evaluation. By default, the
LOAD_FULL, HYPERPARAMS, and IMPORTANCES parameters are set to False due to
a substantial runtime overhead incurred by them.