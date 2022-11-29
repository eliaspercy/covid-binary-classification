"""
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

    python 000810185.py



THE DATA (latestdata.csv)
-------------------------

The program does not require the dataset to be pre-installed, instead it does
require an internet connection because the dataset will automatically be
downloaded, and placed in a local folder entitled 'data'.
If internet access is unavailable but the dataset is installed, please place
the dataset in a folder in the same directory as the Python file entitled
'data', and change the DOWNLOAD_DATASET pramater below to False.



Additional notes
----------------

In terms of runtime, the program will generally take up to 15 minutes to
complete, which entails the training of all three models and the plotting of
various visualisations for exploration and evaluation. By default, the
LOAD_FULL, HYPERPARAMS, and IMPORTANCES parameters are set to False due to
a substantial runtime overhead incurred by them.

"""

# Main imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from typing import (
    List, Union, Tuple, Dict
)

# Data collection
import tarfile
import urllib
import os

# Data tidying and feature engineering
from sklearn.model_selection import StratifiedShuffleSplit
from math import inf
from datetime import date
from collections import Counter
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import DBSCAN, KMeans
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    OneHotEncoder,
)

# Model training
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.pipeline import (
    make_pipeline as make_imba_pipeline,
    Pipeline as ImbaPipeline
)

# Model evaluation
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    matthews_corrcoef,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score
)

# Define F2 score for hyperparameter tuning
from sklearn.metrics import fbeta_score, make_scorer
f2_score = make_scorer(fbeta_score, beta=2)

plt.style.use('fivethirtyeight')

TAR_FILE: str = "latestdata.tar.gz"
DOWNLOAD_URL: str = f"https://github.com/beoutbreakprepared/nCoV2019/blob/" \
                    f"master/latest_data/{TAR_FILE}?raw=true"
DATASET_PATH: str = os.path.join("dataset")
DATASET_NAME: str = "latestdata.csv"
RANDOM_STATE: int = 0

# Disable "chained assignments" warning, because we won't need the original DF
pd.options.mode.chained_assignment = None

# Set the random state to ensure consistent runs
np.random.seed(RANDOM_STATE)

# ------ ADJUSTABLE PARAMETERS ----- #
DOWNLOAD_DATASET: bool = True
HYPERPARAMS: bool = False  # False by default due to runtime
EXPLOATION: bool = True
IMPORTANCES: bool = False  # False by default due to runtime
LOAD_FULL: bool = False  # False by default due to runtime
VERBOSE: bool = True
LINE_LENGTH: int = 50


# -------------------- Dataset Retrieval ------------------------------------ #
"""
The following section of code comprises the initial reading in of the dataset,
first done naively (i.e., the whole dataset is read in without any processing).
After the initial data examination (section subsequent to this), I implemented
a function to read the dataset in by *chunks*, which has a far reduced toll on
memory than naively reading in the entire thing, as less data was in memory at
any one time. Moreover, it allowed me to incorporate some essential
preprocessing - namely, the removal of training examples with a missing
'outcome' value. This was important because 'outcome' was selected to comprise
the binary outputs of the binary classification problem that underscores the
purpose of this program.
"""


def get_covid_dataset(download_url: str = DOWNLOAD_URL,
                      dataset_path: str = DATASET_PATH
                      ) -> None:
    """
    Download the nCoV2019 "latestdata.tar.gz" file from its corresponding
    GitHub URL, extract the "latestdata.csv" dataset and store this in a
    dedicated folder.
    """

    # Retrieve the data from the URL
    filestream = urllib.request.urlopen(download_url)

    # Create a '/data/' directory to store the data if one doesn't exist
    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)

    # Extract the tarfile containing the dataset to the /data/ folder
    with tarfile.open(fileobj=filestream, mode="r|gz") as data_tgz:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(data_tgz, path=dataset_path)


def load_covid_dataset_naive(path: str = DATASET_PATH,
                             name: str = DATASET_NAME
                             ) -> pd.DataFrame:
    """
    Naively load the entire "latestdata.csv" dataset into a Pandas DataFrame.
    """
    csv_path = os.path.join(path, name)
    return pd.read_csv(csv_path)


def process_chunk(chunk: pd.DataFrame,
                  data: pd.DataFrame
                  ) -> pd.DataFrame:
    """
    Remove all missing values from the "outcome" column in the chunk, and
    append the chunk to the DataFrame.
    """
    chunk.dropna(subset=['outcome'], axis=0, inplace=True)
    data = data.append(chunk)
    return data


def load_covid_dataset_in_chunks(path: str = DATASET_PATH,
                                 name: str = DATASET_NAME,
                                 size: int = 500_000
                                 ) -> pd.DataFrame:
    """
    To reduce the burden on memory that loading in the entire dataset naively
    has, load in only the dataset in chunks of size 500000; also, only load in
    rows that are of use - that is, delete the rows containing missing data for
    the 'outcome' feature.
    """
    csv_path = os.path.join(path, name)
    headers = list(pd.read_csv(csv_path, index_col=0, nrows=0).columns)
    data = pd.DataFrame(columns=headers)

    # Iterate through specified size chunks of the data, processing each
    for chunk in pd.read_csv(csv_path, chunksize=size,
                             low_memory=False):
        data = process_chunk(chunk, data)

    del chunk  # delete the "trailing" chunk from memory
    return data


# ----------------- Initial Data Analysis & Preparation --------------------- #
"""
The following section of code comprises the initial examination and analysis of
the dataset after first naively reading the entire dataset into the program.
This includes the visualisation of the proportion of missing values via a
heatmap. Following this, methods for the initial data preparation are
undertaken in order to sufficiently establish the dataset for the problem
domain. This entails the removal of irrelevent features and examples that
contain insufficient important features. Furthermore, the outcome column is
encoded in order to instantiate the binary classification problem.
"""

sns.set_style("whitegrid")


def percentage_missing(column: pd.Series
                       ) -> float:
    """
    Helper function for calculating the percentage of missing values in a
    specified column.
    """
    return column.isnull().sum() * 100 / len(column)


def display_nans(data: pd.DataFrame,
                 filename: str
                 ) -> None:
    """
    Search for NaNs (i.e., missing values) using seaborn's heatmap. This
    provides a convenient illustration of the proportion of missing values for
    each feature.
    """
    plt.figure(figsize=(12, 7))
    hm = sns.heatmap(
        data=data.isnull(),
        cbar=False,
        cmap=sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True)
    )
    bottom, top = hm.get_ylim()
    hm.set_ylim(bottom + 0.5, top - 0.5)
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()


def drop_unhelpful_features(data: pd.DataFrame,
                            verbose: bool = True
                            ) -> pd.DataFrame:
    """
    Drop all "obviously unhelpful" columns; upon brief analysis/visualisation
    of the features in the dataset, certain features offer no assistance as
    regards solving the problem, or contain substantially many missing values.
    """
    if verbose:
        print("Percentages of non-missing values in initial data.")
        print("-" * LINE_LENGTH)
        for col in data.columns:
            percentage = round(percentage_missing(data[col]), 3)
            print(f'{col}: {str(100 - percentage)}%')
        print("-" * LINE_LENGTH)
        print("")

    # Drop all columns that are over 90% missing, or are unlikely to be
    # informative, or are redundant.
    to_drop = [
        "geo_resolution",
        "date_onset_symptoms",
        "date_admission_hospital",
        "symptoms",
        "lives_in_Wuhan",
        "travel_history_dates",
        "travel_history_location",
        "reported_market_exposure",
        "additional_information",
        "chronic_disease",
        "source",
        "sequence_available",
        "date_death_or_discharge",
        "notes_for_discussion",
        "location",
        "admin3",
        "admin2",
        "admin1",
        "country_new",
        "admin_id",
        "data_moderator_initials",
        "ID",
    ]
    data.drop(labels=to_drop, axis=1, inplace=True)
    return data


def analyse_age_and_sex_features(data: pd.DataFrame
                                 ) -> None:
    """
    Gather statistics for the age and sex features: in particular, illustrate
    the amount by which the present (i.e., non-missing) values for each over-
    lap.
    """
    print("Age/sex overlap")
    print("-" * LINE_LENGTH)
    vc = data[~data['sex'].isnull()]['age'].isnull().value_counts()
    overlap = 100 * vc[False] / (vc[True] + vc[False])
    print(vc)
    print(f"Percentage overlap: {overlap}")
    print("-"*LINE_LENGTH + '\n')


def reduce_by_age_and_sex(data: pd.DataFrame,
                          ) -> pd.DataFrame:
    """
    Drop all rows that contain missing values for *both* age and sex. This is
    done in order to reduce the dataset such that it has sufficient values for
    all relevant features.
    """
    data.dropna(subset=['age', 'sex'], how='all', inplace=True)
    return data


def reduce_by_age(data: pd.DataFrame
                  ) -> pd.DataFrame:
    """
    Remove all training examples with missing age value, in order to increase
    the overall relevence of the dataset
    """
    data = data[~data['age'].isnull()]
    return data


def encode_outcomes(data: pd.DataFrame
                    ) -> pd.DataFrame:
    """
    This function is for establishing the binary classification problem. By
    analysisng the value counts of the outcome column, we can compose two sets
    of synonyms for each of "safe" and "severe", and these will
    represent our binary classification values. This function will encode all
    words implying that the patient recovered as 0 and all words implying that
    the patient died, or was placed in critical condition, as 1.
    """
    print(data['outcome'].unique())

    # Manually define feature values remotely synonymous with 'safe' and
    # 'severe' in sets
    safe = {
        'recovered',
        'discharge',
        'alive',
        'stable',
        'stable condition',
        'discharged',
        'discharge from hospital',
        'released from quatantine',
        'hospitalized',
        'not hospitalized'
    }
    severe = {
        'deceased',
        'died',
        'death',
        'dead',
        'severe',
        'severe illness',
        'unstable',
        'critical',
        'critical condition'
    }

    # Map any synonyms of "safe" or "severe" to 0.0 or 1.0 respectively
    # to enforce a binary classification problem. Set unknowns to np.nan, in
    # essence marking them for removal
    def map_outcome(outcome: Union[str, np.float64]
                    ) -> float:
        if outcome in safe:
            return 0.0
        if outcome in severe:
            return 1.0
        return np.nan

    data["outcome"] = data["outcome"].apply(lambda s: map_outcome(s.lower()))
    data.dropna(subset=["outcome"], inplace=True)  # Remove all "unknowns"
    return data


def establish_data(data: pd.DataFrame,
                   verbose: bool = True,
                   age_and_sex: bool = False
                   ) -> pd.DataFrame:
    """
    Function for establishing the dataset for use in the program, by encoding
    the outcomes and reducing the size of the data in order to maximise the
    proportion of existing relevent features. This process ensures that the
    dataset is a suitable representation of the problem domain, and can be
    considered to effectively be the 'initial' dataset following this.
    """
    print(f"\nDataset size prior to establishment: {len(data)}")

    # Drop all clearly unhelpful or simply insufficient features from the data
    data = drop_unhelpful_features(data, verbose)
    if verbose:
        analyse_age_and_sex_features(data)

    # Reduce the dataset by removing columns with missing values for important
    # features - by default only drop examples with missing ages, but can be
    # toggled to keep those with available ages, which will entail further
    # imputing later
    if age_and_sex:
        data = reduce_by_age_and_sex(data)
    else:
        data = reduce_by_age(data)

    # Establish binary classification problem
    data = encode_outcomes(data)

    print(f"\nDataset size after establishment: {len(data)}")
    return data


# ---------------- Obtain the Training and Testing Sets --------------------- #
"""
In this section, the dataset is split into its respective training and testing
sets. Due to the intuitive predictive significance of the age feature, and the
general importance of maintaining an even proportion of the age groups present
due to the human-centric dataset, stratified sampling is incorporated when
splitting the data. The size of the training set is 80% of the original data,
with the remaining 20% going to the testing set - no validation set is created
as cross-fold validation is used later to tune the hyper-parameters and evade
overfitting, which does not necessitate a validation set. Crucially, the
procedure of train/test set splitting is undertaken early on, prior to any of
the preprocessing methods (excluding the data establishment methods from above)
or data exploration visualisations to ensure that overfitting is evaded.
"""


def tidy_age(data: pd.DataFrame,
             ) -> pd.DataFrame:
    """
    In order to handle the age column, the values must first be converted to a
    consistent numerical type. In particular, this function serves to convert
    all age values from inconsistent string formats to consistent int values.
    For ages that are presented as ranges between two ages, an average of the
    two ages is computed and used.
    """
    def map_age(age: Union[str, np.float64]
                ) -> Union[int, np.float64]:
        try:
            try:
                # Try to convert the age to its nearest integer
                return round(float(age))
            except ValueError:
                # Try to split age ranges by the '-' and return the average.
                return round(np.average([int(x) for x in str(age).split('-')]))
        except ValueError:
            return np.nan

    data['age'] = data['age'].apply(map_age)
    return data


def create_age_categories(data: pd.DataFrame,
                          ) -> pd.DataFrame:
    """
    Create a new feature containing age ranges, for use in stratified sampling.
    This essentially entails placing the numerical age feature into categories
    based on the age range they fall under. In addition for use in the
    stratified sampling, which is the original purpose of these procedure, this
    is both convenient for data analysis and will be used later down the line
    as a hyperparameter for the training models.
    """
    age_ranges = ["0-14", "15-29", "30-44", "45-59", "60-74", "75+"]
    data["age_categories"] = pd.cut(
        data["age"],
        bins=[0, 15, 30, 45, 60, 75, inf],
        labels=age_ranges,
        ordered=True
    ).values.add_categories("NaN")
    data["age_categories"].fillna("NaN", inplace=True)
    return data, age_ranges


def display_age_cat_histogram(data: pd.DataFrame,
                              ) -> None:
    """
    Histogram to display the distribution of the age categories, allowing us to
    gain an intuitive insight into the success of the stratified sampling.
    """
    plt.figure(figsize=(8, 5))
    cat_order = ["0-14", "15-29", "30-44", "45-59", "60-74", "75+"]
    sns.countplot(
        x="age_categories",
        data=data,
        order=cat_order,
        palette=sns.color_palette("crest")
    )
    plt.show()


def stratified_train_test(data: pd.DataFrame,
                          test_size: float = 0.2
                          ) -> Tuple[pd.DataFrame, ...]:
    """
    Utilising stratified sampling, split the dataset into respective train and
    test sets whilst maintaining a consistent distribution of age values in
    each.
    """

    # Instantiate a class for handling the stratified sampling, using a
    # specified random state to ensure that the split is consistent in
    # subsequent runs.
    split = StratifiedShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=RANDOM_STATE
    )

    # Obtain two sets of indices for each of the train and test sets
    stratified_split = split.split(data, data["age_categories"])

    # Construct the train and test sets using the aforementioned indices
    for train_index, test_index in stratified_split:
        stratified_train_set = data.iloc[train_index]
        stratified_test_set = data.iloc[test_index]

    return stratified_train_set, stratified_test_set


def obtain_train_test_sets(data: pd.DataFrame,
                           verbose: bool = True
                           ) -> Tuple[pd.DataFrame, ...]:
    """
    Container function for the methodology behind obtaining the train and test
    sets from the data, involving calls to the functions for generating the
    age categories feature, the stratified sampling, and optionally visualising
    the age category distributions within each of the datasets.
    """

    # Tidy the age column, which initially comprises strings that have a
    # numerical equivalent
    data = tidy_age(data)

    # Place the now-tidied ages into distinct categories
    data, _ = create_age_categories(data)
    if verbose:
        display_age_cat_histogram(data)

    # Undertake stratified sampling to split the dataset into a testing and
    # training set, where the training set will comprise 80% of the original
    # data and the testing set will comprise the remaining 20%
    test_train_sets = stratified_train_test(data)

    # verifying the success of the stratified train/test split by comparing
    # histograms showcasing the distribution of age category values in each of
    # the sets.
    if verbose:
        for data_set in test_train_sets:
            display_age_cat_histogram(data_set)

    # For the time being, remove this newly appended age category feature. The
    # retention of it will be left as a hyperparameter.
    for data_set in test_train_sets:
        data_set.drop("age_categories", axis=1, inplace=True)

    return test_train_sets


# ------------------ Data Exploration and Visualisation --------------------- #
"""
The following section comprises the exploration of the training set through the
use of textual statistics and visualisations. Feature engineering and data
tidying was trialed here in search for corellations and patterns. Many of the
feature engineering and data tidying methods devised in this section were later
reused for the official data preprocessing (i.e., the next section). Data
exploration was done on a copy of the training set to avoid potentially
constructing methods that would lead to overfitting if the testing data were
visible, and also to avoid accidently editing the original training data.
"""

# ----- Methods for visualising the data


def visualise_geographically(data: pd.DataFrame
                             ) -> None:
    """
    Visualise the data geographically, this time incorporating the geographic
    variations in outcome
    """
    plt.figure(figsize=(7, 4))
    sns.scatterplot(
        data=data,
        x='longitude',
        y='latitude',
        hue='outcome',
        alpha=0.2,
    )
    plt.show()


def visualise_corr(data: pd.DataFrame,
                   feature: str,
                   outcome: str = 'outcome'
                   ) -> None:
    """
    For a given feature, visualise the correlation with outcome via a scatter
    plot, and print the corresponding Matthew's correlation coefficient value
    """
    print(matthews_corrcoef(data[feature], data[outcome]))
    plt.figure(figsize=(8, 8))
    plt.scatter(data[feature], data["outcome"],
                marker='+', color="red", alpha=0.1)
    plt.show()


def show_outcome_proportions_per_age(data: pd.DataFrame
                                     ) -> None:
    """
    Function for obtaining the proportions of severe cases per age categories
    """
    plt.figure(figsize=(8, 7))
    data_age_cats, ranges = create_age_categories(data)
    sns.countplot(
        data=data_age_cats,
        x='age_categories',
        hue='outcome',
        order=ranges
    )
    plt.show()


def violinplot_age_outcome(data: pd.DataFrame
                           ) -> None:
    """
    Using violin-plots, visualise the distribution of the age feature for each
    outcome (0 and 1) in order to gain intuition as regards the correlation
    between age and outcome.
    """
    plt.figure(figsize=(8, 7))
    sns.violinplot(
        y=data['outcome'], x=data['age'], orient='h'
    )
    plt.show()


def pairplot(data: pd.DataFrame
             ) -> None:
    """
    Produce a pairplot between all features and the outcome in order to detect
    any unexpected corellations.
    """
    plt.figure(figsize=(8, 7))
    display_data = data.dropna()
    sns.pairplot(display_data, hue='outcome', diag_kind='kde')
    plt.show()


def get_outcome_percentages(data: pd.DataFrame
                            ) -> None:
    """
    Function for obtaining the ratio of outcomes, i.e. negative:positive
    """
    neg = len(data[data["outcome"] == 0])
    pos = len(data[data["outcome"] == 1])
    tot = len(data)
    percentage = f"{100*pos/tot:.3f}%"
    ratio = f"{neg/pos:.1f}:{1}"
    print("\nExamining outcomes")
    print("-"*LINE_LENGTH)
    print(f"Percentage of positive outcomes:       {percentage}")
    print(f"Ratio of outcomes (negative:positive): {ratio}\n")


def date_distribution(data: pd.DataFrame
                      ) -> None:
    """
    Visualise the distribution of dates by illustrating the densities.
    """
    sns.displot(
        data=data,
        x="days_since_start",
        hue="outcome",
        kind="kde",
    )
    bottom, top = plt.ylim()
    plt.ylim(bottom - 1e-3, top)
    plt.figure(figsize=(8, 4))
    plt.show()


def age_chronic_severity(data: pd.DataFrame
                         ) -> None:
    """
    Illustrate any correltaion between the age of the patient, the severity,
    and the presence of a chronic disease.
    """
    plt.figure(figsize=(10, 5))
    sns.stripplot(
        data=data,
        x="age", y="outcome_cat", hue="chronic_disease_binary",
        dodge=True
    )
    plt.show()


# ----- Methods for data cleaning and feature engineering


DAY_ZERO: date = date(2019, 12, 1)  # I.e., the first recorded Covid-19 case


def cvt2_days_since_start(data: pd.DataFrame,
                          date_col: str = 'date_confirmation',
                          day0: date = DAY_ZERO
                          ) -> pd.Series:
    """
    In order to actually reason about the date/time data present in the
    dataset, it must first be converted into some numerical encoding. This
    function serves to transform the "date confirmation" feature into a new
    feature, one that is more meaningful to a computer program. Specifically,
    the Python datetime library is utilised to convert the string data present
    in this column into a Date object, and using this the number of days, as an
    integer, is calculated from the date of the first reported case (defined in
    the DAY_ZERO constant above).
    """

    # Helper function for mapping strings to corresponding date objects
    def string2date(d: str
                    ) -> date:
        ns = [int(n) for n in d.split('.')]
        return date(ns[2], ns[1], ns[0])

    # Helper function for calculating the number of days since DAY_ZERO
    def map_dates(s: Union[str, np.float64]
                  ) -> float:
        try:
            delta = [string2date(d) for d in s.split('-')][0] - day0
            return delta.days
        except AttributeError:
            return np.nan

    return data[date_col].apply(map_dates)


def encode_binaries(data: pd.DataFrame,
                    features: list = None,
                    ) -> pd.DataFrame:
    """
    Encode the binary features
    """
    if features is None:
        features = ['travel_history_binary',
                    'chronic_disease_binary']
    for feature in features:
        data[feature] = data[feature].astype(float)  # True -> 1, False -> 0
        data[feature].fillna(data[feature].mode().iloc[0], inplace=True)
    return data


def create_count_per_day(data: pd.DataFrame
                         ) -> pd.DataFrame:
    """
    Engineer a new feature: "count per day". This will correspond with the
    frequency of a certain date in the dataset, so that the feature illustrates
    the coronavirus rate at that time period
    """
    count_per_day = data['days_since_start']\
        .map(data['days_since_start'].value_counts())
    data['count_per_day'] = count_per_day
    return data


def remove_country_outliers(data: pd.DataFrame
                            ) -> pd.DataFrame:
    """
    In order to evade geographic overfitting, remove any underrepresented
    country in the dataset.
    """

    # Define a set of sufficiently represented countries
    acceptable = set(data['country'].value_counts()[:5].index.tolist())
    """
    Includes:
        "India", "Philippines", "China", "Ethiopia", and "Singapore"
    """

    # Remove all countries not in the aforementioned set from the dataset
    data = data[data["country"].isin(acceptable)]
    return data


def create_severe_rate(data: pd.DataFrame
                       ) -> pd.DataFrame:
    """
    Create a new feature for the death count on each day
    """
    only_deaths = data[data['outcome'] == 1]
    death_counts = data['days_since_start']\
        .map(only_deaths['days_since_start'].value_counts())
    data['severity_rate'] = death_counts
    return data


def remove_age_outliers(data: pd.DataFrame
                        ) -> pd.DataFrame:
    """
    Remove any age outliers in the positive class to mitigate harmful impact on
    trained models.
    """

    # Separate dataframe by outcomes
    data_neg = data[data["outcome"] == 0]
    data_pos = data[data["outcome"] == 1]

    # Do outlier removal
    data_neg = data_neg[
        data_neg["age"].between(
            data_neg["age"].quantile(.02),
            data_neg["age"].quantile(.98)
        )
    ]
    data_pos = data_pos[
        data_pos["age"].between(
            data_pos["age"].quantile(.02),
            data_pos["age"].quantile(.98)
        )
    ]

    # Recombine dataset and return
    data = data_pos.append(data_neg)
    return data


def data_exploration(data: pd.DataFrame,
                     verbose: bool = False,
                     show_pairplot: bool = False
                     ) -> None:
    """
    Container function for the various data exploration methods. Exploration is
    done on a copy of the training set: only the training set is used to avoid
    feature engineering that leads to unintentional overfitting.
    """

    print("Exploring dataset...\n")

    # Create a copy of the dataset so as to not distort it's content whilst
    # exploring
    exploration_data = data.copy()
    # print(exploration_data)

    exploration_data['outcome_cat'] = (
        exploration_data['outcome'].astype('category')
    )

    get_outcome_percentages(exploration_data)

    # Encode the binary features to 0 and 1, for the sake of exploration
    exploration_data = encode_binaries(exploration_data)

    # Search for patterns!
    age_chronic_severity(exploration_data)
    exploration_data = remove_age_outliers(exploration_data)

    # Show the proportions of outcomes by age groups
    violinplot_age_outcome(exploration_data)
    show_outcome_proportions_per_age(exploration_data)

    exploration_data['days_since_start'] = cvt2_days_since_start(
        exploration_data
    )
    exploration_data.drop(labels=["date_confirmation"], axis=1)

    # Visualise the dates
    date_distribution(exploration_data)

    visualise_geographically(data)
    print(data["country"].value_counts())

    # Encode the longitude and latitude
    exploration_data["geo"] = GeoHandler().fit_transform(
        exploration_data[["longitude", "latitude"]]
    )
    exploration_data.drop(
        labels=["longitude", "latitude"], axis=1, inplace=True
    )

    # Drop too sparse geographical features, subsumed by the new geo feature
    exploration_data.drop(
        labels=["city", "country", "province"], axis=1, inplace=True
    )

    exploration_data["sex"].replace({"female": 0, "male": 1}, inplace=True)
    exploration_data.dropna(inplace=True)

    if show_pairplot:
        pairplot(exploration_data)

    print("Finished data exploration.")


# --------------------- Prepare for Machine Learning ------------------------ #
"""
The following section comprises the final data preprocessing stage. Here, many
methods devised in the prior section have been re-used in the form of custom
transformers, etc., which are then incorporated into a column transformer to
address the various different requirements of each feature. The transformers
have been designed to facilitate hyperparameter tuning with ease.
"""


def data_preparation(data: pd.DataFrame,
                     display_geog: bool = False
                     ) -> pd.DataFrame:
    """
    Container method for the essential data cleaning operations that must be
    undertaken prior to the machine learning pipelines. Here, various methods
    and functions developed during dataset exploration are reused, whilst other
    methods that don't pertain to any deletion of data are saved for use within
    custom transformers later.
    """

    # Encode values into numerical format
    data['days_since_start'] = cvt2_days_since_start(
        data, 'date_confirmation'
    )
    data.drop("date_confirmation", axis=1, inplace=True)

    # Remove geographic outliers
    data = remove_country_outliers(data)

    # Visualise the change in geographic data, if toggled
    if display_geog:
        visualise_geographically(data)

    # Encode the binary features into 0 & 1
    data = encode_binaries(data)

    # Create the "death rate" feature (this can't be done in a pipeline as it
    # requires the outcome column)
    data = create_severe_rate(data)

    # Remove age outliers
    data = remove_age_outliers(data)

    # Drop the redundent geographic features (subsumed by longitude/latitude)
    data.drop(labels=["country", "province", "city"], axis=1, inplace=True)

    return data


# ---- Construction of custom transformer classes for feature engineering


class DateHandler(BaseEstimator, TransformerMixin):
    """
    Custom Transformer class for creating an attribute pertaining to the "rate"
    of coronavirus on a given date - i.e., the number of instances of each
    given date in the dataframe. Will be a Hyper param due to uncertainty.
    """

    def __init__(self,
                 create_day_count: bool = True
                 ) -> None:
        self.create_day_count = create_day_count

    def fit(self,
            X: pd.DataFrame,
            y: pd.DataFrame = None
            ) -> 'DateHandler':
        return self

    def transform(self,
                  X: pd.DataFrame
                  ) -> pd.DataFrame:

        if self.create_day_count:
            X = create_count_per_day(X)

        return X


class AgeHandler(BaseEstimator, TransformerMixin):
    """
    Custom transformer for age category creation
    """

    def __init__(self,
                 make_age_categories: bool = True,
                 num_age_categories: int = 4,
                 max_age: int = 80
                 ) -> None:
        self.make_age_categories = make_age_categories
        self.num_age_categories = make_age_categories
        self.max_age = max_age

    def fit(self,
            X: pd.DataFrame,
            y: pd.DataFrame = None
            ) -> 'AgeHandler':
        return self

    def transform(self,
                  X: pd.DataFrame
                  ) -> np.c_:
        X = tidy_age(X)
        if self.make_age_categories:
            age_ranges = [i for i in range(self.num_age_categories)]
            r = self.max_age // self.num_age_categories
            bins = [i*r for i in range(self.num_age_categories)] + [np.inf]

            X["age_categories"] = pd.cut(
                X["age"],
                bins=bins,
                labels=age_ranges,
                ordered=True
            )
            X["age_categories"].fillna(0, inplace=True)

            X.drop(labels=["age"], axis=1, inplace=True)
        return X


class GeoHandler(BaseEstimator, TransformerMixin):
    """
    Custom transformer for utilising DBSCAN to cluster the geographic data
    (i.e., the "longitude" and "latitude" features)
    """

    def __init__(self,
                 ) -> None:
        pass

    def fit(self,
            X: pd.DataFrame,
            y: pd.DataFrame = None
            ) -> 'GeoHandler':
        return self

    def transform(self,
                  X: pd.DataFrame
                  ) -> np.c_:
        coords = X.to_numpy()
        db = DBSCAN(
            eps=1.5 / 6.3710088e3,
            min_samples=1,
            algorithm='ball_tree',
            metric='haversine'
        ).fit(np.radians(coords))

        return np.array([[label] for label in db.labels_])


def create_transformer(disclude_features: list = None,
                       age_cats: bool = False,
                       k_clusters: int = 12
                       ) -> ColumnTransformer:
    """
    Function for creating a collection of Pipelines, each for performing
    specific preprocessing operations on different columns (for instance, the
    categorical features must be handled differently to the numerical
    features). These Pipelines are contained in a ColumnTransformer for
    convenience.
    """
    if disclude_features is None:
        disclude_features = []

    all_categorical = ["sex"]
    all_numerical = ["age", "days_since_start", "severity_rate",
                     "longitude", "latitude"]
    all_binary = ["chronic_disease_binary", "travel_history_binary"]

    categorical_features = [f for f in all_categorical
                            if f not in disclude_features]
    numerical_features = [f for f in all_numerical
                          if f not in disclude_features]
    binary_features = [f for f in all_binary
                       if f not in disclude_features]

    cat_transformer = Pipeline(steps=[
        ("cat_imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])

    num_transformer = Pipeline(steps=[
        ("covid_rate", DateHandler()),
        ("num_imputer", SimpleImputer(strategy="median")),
    ])

    date_transformer = Pipeline(steps=[
        ("num_imputer", SimpleImputer(strategy="median")),
        ("cluster", KMeans(n_clusters=k_clusters)),
    ])

    age_transformer = Pipeline(steps=[
        ("age_handle", AgeHandler()),
        ("encoder", OneHotEncoder(handle_unknown='ignore'))
    ])

    geo_transformer = Pipeline(steps=[
        ("geo_handler", GeoHandler()),
    ])

    steps = [
        ("categorical", cat_transformer, binary_features+categorical_features),
        ("numerical", num_transformer, numerical_features),
    ]

    if "days_since_start" not in disclude_features:
        steps.append(("dates", date_transformer, ["days_since_start"]))

    if "geography" not in disclude_features:
        steps.append(("geo", geo_transformer, ["longitude", "latitude"]))

    if age_cats:
        steps.append(("ages", age_transformer, ["age"]))

    ct = ColumnTransformer(steps, remainder="drop")

    return ct


# ------------------------ Machine Learning --------------------------------- #
"""
The following section comprises the machine learning stage of the program.
Three different classification models are trained - namely, Logistic
Regression, Random Forest Classification, and Support Vector Classification.
For each, there are four functions. First, for obtaining the basic variant of
the model with no specific hyperparameter tuning applied, secondly for applying
tuned hyperparameters to the models with additional enhancements for handling
imbalanced data (i.e., SMOTE), third for actually tuning the hyperparameters
via cross fold validation (with either GridSearchCV or RandomizedSearchCV,
dependening of the time complexity of the model in question), and finally a
container function for each holding the three aforementioned methods.
"""


def split_X_y(data: pd.DataFrame,
              ) -> pd.DataFrame:
    """
    Seperate the features from the outcome
    """
    split_data = (data.drop("outcome", axis=1), data["outcome"].copy())
    return split_data


def get_outcome_ratio(outcomes: List[int]
                      ) -> Dict[int, float]:
    """
    Obtain the ratio of negative and positive outcomes for use as class weights
    """
    counts = Counter(outcomes)
    return {
        0: counts[0]/counts[1],
        1: 1
    }


# --------------------------------------------------------------------------- #
# --------------------- 1. Logistic Regression ------------------------------ #


def get_basic_logistic_regression_model(transformer: ColumnTransformer
                                        ) -> Pipeline:
    """
    Return the basic logistic regression model for the dataset, without any
    consideration with respect to hyperparameter tuning
    """
    model = Pipeline([
        ('col_trans', transformer),
        ('scaler', StandardScaler(with_mean=True)),
        ('log_reg', LogisticRegression())
    ])
    return model


def get_tuned_logistic_regression_model(transformer: ColumnTransformer
                                        ) -> Pipeline:
    """
    This function returns the logistic regression model using the parameters
    that have been tuned via cross validation.
    """
    imba_pipeline = make_imba_pipeline(
        BorderlineSMOTE(
            sampling_strategy=0.5,
            random_state=RANDOM_STATE
        ),
        LogisticRegression(
            random_state=RANDOM_STATE,
            max_iter=50000,
            C=0.23357214690901212,
            penalty='l2',
            solver='saga',
            dual=False
        )
    )

    model = Pipeline([
        ('col_trans', transformer),
        ('scaler', StandardScaler(with_mean=True)),
        ('pca', PCA(n_components=11)),
        ('log_reg', imba_pipeline)
    ])
    return model


def logistic_regression_cross_validation(transformer: ColumnTransformer
                                         ) -> GridSearchCV:
    """
    k-Fold cross validation for logistic regression.
    """
    hyperparams = {
        'upsample__sampling_strategy': [0.5, 0.55, 0.6],
        "pca__n_components": [11, 12, 13],
        'lr__penalty': ['l1', 'l2', 'elasticnet'],
        'lr__C': np.logspace(-4, 4, 20),
        'lr__solver': ['lbfgs', 'liblinear', 'saga'],
        "lr__intercept_scaling": [1, 2, 4, 5],
        'lr__class_weight': [None, {0: 1, 1: 10}, {0: 1, 1: 100}, 'balanced'],
    }
    imba_pipeline = ImbaPipeline([
        ('pca', PCA()),
        ("upsample", BorderlineSMOTE(random_state=RANDOM_STATE)),
        ("lr", LogisticRegression(max_iter=30000, random_state=RANDOM_STATE))]
    )
    grid = Pipeline([
        ('col_trans', transformer),
        ('std_scaler', StandardScaler(with_mean=True)),
        ('cross_val', GridSearchCV(
            estimator=imba_pipeline,
            param_grid=hyperparams,
            scoring=f2_score,
            n_jobs=-1,
            verbose=3
        ))
    ])
    return grid


def logistic_regression(train_data: Tuple[pd.DataFrame, ...],
                        get_best_params: bool = False,
                        get_basic_model: bool = False
                        ) -> Pipeline:
    """
    Container function for managing the logistic regression methods
    """
    train_X, train_y = train_data
    transformer = create_transformer([])
    if get_best_params:
        model = logistic_regression_cross_validation(transformer)
        model.fit(train_X, train_y)
        print(model['cross_val'].best_params_)
    elif get_basic_model:
        model = get_basic_logistic_regression_model(transformer)
        model.fit(train_X, train_y)
    else:
        st = time.time()
        model = get_tuned_logistic_regression_model(transformer)
        model.fit(train_X, train_y)
        print(f"Time elapsed in training LR: {time.time() - st:.2f} seconds.")
    return model


# Tuned hyperparameters for Logistic Regression
""" F2 Score
{
 'upsample__sampling_strategy': 0.5,
 'pca__n_components': 11,
 'lr__solver': 'saga',
 'lr__penalty': 'l2',
 'lr__C': 0.23357214690901212
}
"""


# --------------------------------------------------------------------------- #
# ------------------------- 2. Random Forest -------------------------------- #


def get_basic_random_forest_model(transformer: ColumnTransformer
                                  ) -> Pipeline:
    """
    Return the initial random forest classifier prior to any hyperparameter
    tuning, or the introduction of any ehancement measures.
    """
    model = Pipeline([
        ('col_trans', transformer),
        ('std_scaler', StandardScaler(with_mean=True)),
        ('forest', RandomForestClassifier())
    ])
    return model


def get_tuned_random_forest_model(transformer: ColumnTransformer
                                  ) -> Pipeline:
    """
    Obtain the random forest model while utilising the parameters optimised
    via cross validation, and make use of dimensionality reduction and SMOTE.
    """
    imba_pipeline = make_imba_pipeline(
        BorderlineSMOTE(
            sampling_strategy=0.5,
            random_state=RANDOM_STATE
        ),
        RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_estimators=800,
            min_samples_split=10,
            min_samples_leaf=4,
            max_features="auto",
            max_depth=80,
            bootstrap=True,
            criterion="entropy",
            class_weight="balanced_subsample"
        )
    )

    rfc_pl = Pipeline([
        ('col_trans', transformer),
        ('scaler', MinMaxScaler(feature_range=(0, 1))),
        ('pca', PCA(n_components=11)),
        ('rfc', imba_pipeline)
    ])
    return rfc_pl


def random_forest_cross_validation(transformer: ColumnTransformer
                                   ) -> GridSearchCV:
    """
    Cross validation for the random forest classifier, primarily for the sake
    of tuning the hyperparameters
    """
    hyperparams = {
        "upsample__sampling_strategy": [0.3, 0.5, 0.8],
        "pca__n_components": [3, 6, 11, 13],
        "rfc__bootstrap": [True, False],
        "rfc__max_depth": [30, 60, 90, None],
        "rfc__max_features": ["auto", "log2"],
        "rfc__min_samples_leaf": [1, 2, 4],
        "rfc__min_samples_split": [2, 5, 10],
        "rfc__n_estimators": [100, 400, 800, 1000, 1600],
        "rfc__class_weight": ['balanced_subsample', "balanced"],
        "rfc__criterion": ["entropy", "gini"]
    }
    imba_pipeline = ImbaPipeline([
        ("upsample", BorderlineSMOTE()),
        ('pca', PCA()),
        ("rfc", RandomForestClassifier(random_state=RANDOM_STATE))
    ])

    grid = Pipeline([
        ('col_trans', transformer),
        ('scaler', MinMaxScaler(feature_range=(0, 1))),
        ('cross_val', GridSearchCV(
            estimator=imba_pipeline,
            param_grid=hyperparams,
            n_iter=20,
            cv=3,
            scoring=f2_score,
            n_jobs=-1,
            verbose=3
        ))
    ])
    return grid


def random_forest_classifier(train_data: Tuple[pd.DataFrame, pd.DataFrame],
                             get_best_params: bool = False,
                             get_basic_model: bool = False
                             ) -> None:
    """
    Container function for the random forest classifier and its associated
    methods
    """
    train_X, train_y = train_data
    transformer = create_transformer(['sex'], age_cats=True)
    if get_best_params:
        model = random_forest_cross_validation(transformer)
        model.fit(train_X, train_y)
        print(model['cross_val'].best_params_)
    elif get_basic_model:
        model = get_basic_random_forest_model(transformer)
        model.fit(train_X, train_y)
    else:
        st = time.time()
        model = get_tuned_random_forest_model(transformer)
        model.fit(train_X, train_y)
        print(f"Time elapsed in training RFC: {time.time() - st:.2f} seconds.")
    return model


# Tuned hyperparameters for Random Forest
""" Scorer: F2 score
{
 'upsample__sampling_strategy': 0.5,
 'rfc__n_estimators': 800,
 'rfc__min_samples_split': 10,
 'rfc__min_samples_leaf': 4,
 'rfc__max_features': 'auto',
 'rfc__max_depth': 80,
 'rfc__criterion': 'entropy',
 'rfc__class_weight': 'balanced_subsample',
 'rfc__bootstrap': True,
 'pca__n_components': 11
}
Scaling: MinMax
"""

# --------------------------------------------------------------------------- #
# -------------------- 3. Support Vector Classifier ------------------------- #


def get_basic_support_vector_classifier_model(transformer: ColumnTransformer
                                              ) -> Pipeline:
    """
    Return the basic support vector classifier for the dataset, without any
    consideration with respect to hyperparameter tuning
    """
    model = Pipeline([
        ('col_trans', transformer),
        ('std_scaler', StandardScaler(with_mean=True)),
        ('svc', SVC(probability=True))
    ])
    return model


def get_tuned_support_vector_classifier_model(transformer: ColumnTransformer
                                              ) -> Pipeline:
    """
    Use the optimal hyperaparameters acquired from cross validation to return
    the tuned support vector classifier model.
    """
    imba_pipeline = make_imba_pipeline(
        BorderlineSMOTE(
            sampling_strategy=1,
            random_state=RANDOM_STATE
        ),
        SVC(
            probability=True,
            C=0.5,
            gamma='scale',
            class_weight='balanced',
            kernel='rbf',
            random_state=RANDOM_STATE
        )
    )
    support_vector_pipeline = Pipeline([
        ('col_trans', transformer),
        ('scaler', StandardScaler(with_mean=True)),
        ('pca', PCA(n_components=11, random_state=RANDOM_STATE)),
        ('svc', imba_pipeline)
    ])
    return support_vector_pipeline


def support_vector_cross_validation(transformer: ColumnTransformer
                                    ) -> GridSearchCV:
    """
    Cross fold validation for the support vector classifier, with the
    introduction also of SMOTE and PCA.
    """
    hyperparams = {
        'upsample__sampling_strategy': [0.2, 0.5, 0.9, 1],
        'pca__n_components': [3, 6, 10, 13],
        'svc__C': [5*10**i for i in range(-4, 4)],
        'svc__class_weight': [None, 'balanced'],
        'svc__gamma': ['scale', 'auto'],
        'svc__degree': [1, 3],
    }

    imba_pipeline = ImbaPipeline([
        ('pca', PCA()),
        ("upsample", BorderlineSMOTE(random_state=RANDOM_STATE)),
        ("svc", SVC(probability=True))
    ])

    grid = Pipeline([
        ('col_trans', transformer),
        ('scaler', MinMaxScaler(feature_range=(0, 1))),
        ('cross_val', RandomizedSearchCV(
            estimator=imba_pipeline,
            param_distributions=hyperparams,
            scoring=f2_score,
            n_jobs=-1,
            verbose=3
        ))
    ])
    return grid


def support_vector_classifier(train_data: Tuple[pd.DataFrame, pd.DataFrame],
                              get_best_params: bool = False,
                              get_basic_model: bool = False
                              ) -> None:
    """
    Container for the support vector classifer.
    """
    train_X, train_y = train_data
    transformer = create_transformer([])
    if get_best_params:
        model = support_vector_cross_validation(transformer)
        model.fit(train_X, train_y)
        print(model['cross_val'].best_params_)
    elif get_basic_model:
        model = get_basic_support_vector_classifier_model(transformer)
        model.fit(train_X, train_y)
    else:
        st = time.time()
        model = get_tuned_support_vector_classifier_model(transformer)
        model.fit(train_X, train_y)
        print(f"Time elapsed in training SVC: {time.time() - st:.2f} seconds.")
    return model


# Tuned hyperparameters for SVC
""" Scorer: F2
{
 'upsample__sampling_strategy': 1,
 'svc__C': 0.5,
 'svc__class_weight': 'balanced',
 'svc__gamma': 'scale',
 'pca__n_components': 11
}
Scaling: Standard
"""


# --------------------------------------------------------------------------- #


# -------------------------- Analyse Results -------------------------------- #
"""
The following section comprises various methodology for evaluating the relative
performances of each of the trained models. Much attention has been placed
regarind this aspect of the program due to contrasting connotations of
differing evaluation metrics, particularly for binary classification problems
with imbalanced data, such as this. In addition to simple textual evluation in
the form of printing the accuracy, F1, precision, and recall scores for the
trained models, visual illustrations of the performances is included - i.e.,
precision-recall curves, receiver operating characteristics curves, and
heatmaps showcasing the confusion matrix. This visualisations are key as they
dispense a more insightful exemplar of the relative performances by clearly
indicating the true/false positive rates, the effect of the prediction
threshold on accuracy, and the trade-off in precision and recall.
"""


# Define a function to look at the success of the predictions for any model
def analyse_results_text(model_name: str,
                         model: Pipeline,
                         test_data: Tuple[pd.DataFrame, pd.DataFrame],
                         train_data: Tuple[pd.DataFrame, pd.DataFrame]
                         ) -> None:
    """
    Function for analysing the performance of a model
    """
    test_X, test_y = test_data
    predictions = model.predict(test_X)
    preds_proba = model.predict_proba(test_X)

    print("")
    print(f"Results for {model_name}.")
    print("-"*LINE_LENGTH)
    print(f"Accuracy score:  {accuracy_score(test_y, predictions)}.")
    print(f"F1 score:        {f1_score(test_y, predictions)}.")
    print(f"F2 score:        {fbeta_score(test_y, predictions, beta=2)}.")
    print(f"Precision score: {precision_score(test_y, predictions)}.")
    print(f"Recall score:    {recall_score(test_y, predictions)}.")
    print(f"ROC-AUC score:   {roc_auc_score(test_y, preds_proba[:, 1])}.")

    print("\nClassification Report:")
    print(classification_report(test_y, predictions))
    print("\n Confusion Matrix:"
          "\n([[True Negative,  False Positive]"
          "\n  [False Negative, True Positive ]]\n")
    print(confusion_matrix(test_y, predictions))

    print("\nChecking for overfitting:")
    train_X, train_y = train_data
    print(f"Train recall: {recall_score(train_y, model.predict(train_X))}.")
    print(f"Test recall:  {recall_score(test_y, predictions)}.")


def confusion_matrix_heatmap(model_name: str,
                             trained_model: Pipeline,
                             test_data: Tuple[pd.DataFrame, pd.DataFrame]
                             ) -> None:
    """
    Visualise a detailed confusion matrix for the inputted model, including
    absolute counts and percentages.
    """
    test_X, test_y = test_data

    plt.figure(figsize=(5, 3))

    predictions = trained_model.predict(test_X)
    confusion = confusion_matrix(test_y, predictions)
    confusion_df = pd.DataFrame(
        data=confusion,
        index=["True 0", "True 1"],
        columns=["Predicted 0", "Predicted 1"]
    )
    vis = sns.heatmap(
        data=confusion_df,
        annot=True,
        fmt='d',
        cmap='Blues',
        cbar=False
    )
    bottom, top = vis.get_ylim()
    vis.set_ylim(bottom + 0.5, top - 0.5)
    plt.show()


def illustrate_roc_curves(trained_models: Tuple[Tuple[Pipeline, str], ...],
                          test_data: Tuple[pd.DataFrame],
                          title: str
                          ) -> None:
    """
    This method will take a list or tuple of some trained models and draw the
    respective ROC curves for each on the same plot. This allows us to observe
    a visualisation of the effectiveness of the hyperparameter tuning, and the
    relative success of the trained models.
    """
    test_X, test_y = test_data

    # Make canvas square
    plt.figure(figsize=(7.5, 6))

    # Draw a straight, 45 degree line to represent the "random number
    # generator" output to conttrast with model ROC curves
    sns.lineplot(
        x=[0, 1], y=[0, 1], label="Unskilled Classifier"
    ).lines[0]

    # Assign titles and labels
    plt.title(title)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    # Generate and draw the curves
    for model, name in trained_models:
        predictions = model.predict_proba(test_X)
        score = roc_auc_score(test_y, predictions[:, 1])
        false_pos_rate, true_pos_rate, thresholds = roc_curve(
            test_y, predictions[:, 1]
        )
        sns.lineplot(
            x=false_pos_rate, y=true_pos_rate,
            label=f"{name}, AUC = {score:.4f}"
        )
    plt.legend(loc="lower right")
    plt.show()


def illustrate_pr_curves(trained_models: Tuple[Tuple[Pipeline, str], ...],
                         test_data: Tuple[pd.DataFrame],
                         title: str
                         ) -> None:
    """
    Illustrate precision recall curves to gain further insight into the
    classification performances of the models.
    """
    test_X, test_y = test_data

    # Make canvas square
    plt.figure(figsize=(7.5, 6))

    # Draw a straight line to represent the "random number generator" output
    sns.lineplot(
        x=[0, 1], y=len(test_y[test_y == 1])/len(test_y),
        label="Unskilled Classifier"
    ).lines[0]

    # Assign titles and labels
    plt.title(title)
    plt.xlabel("Recall")
    plt.ylabel("Precision")

    # Generate and draw the curves
    for model, name in trained_models:
        predictions = model.predict_proba(test_X)
        score = average_precision_score(test_y, predictions[:, 1])
        precision, recall, thresholds = precision_recall_curve(
            test_y, predictions[:, 1]
        )
        sns.lineplot(
            x=recall, y=precision,
            label=f"{name}, \nAverage Precision = {score:.4f}"
        )
    plt.legend(loc="upper right")
    plt.show()


def obtain_feature_importances(model: Pipeline,
                               test_data: Tuple[pd.DataFrame, ...],
                               title: str
                               ) -> None:
    """
    Illustrate the permutation importance (per model), for feature evaluation
    """
    test_X, test_y = test_data

    # Get feature names
    features = list(test_X.columns.values)

    # Obtain feature importances
    results = permutation_importance(
        estimator=model,
        X=test_X, y=test_y,
        scoring=f2_score,
        n_repeats=30,
        random_state=RANDOM_STATE
    )

    importances_df = pd.DataFrame(
        [{'feature': features[i],
          'importances_mean': abs(results['importances_mean'][i]),
          'importances_std': results['importances_std'][i]}
         for i in range(len(features))
         ]
    )

    plt.figure(figsize=(7, 10))

    plt.title(title)
    plt.xlabel("Features")
    plt.ylabel("Relative Mean Importance")

    bp = sns.barplot(
        data=importances_df,
        x='feature', y='importances_mean',
        palette=sns.color_palette("crest"),

    )

    # Orient and position x-labels for the sake of clarity
    bp.set_xticklabels(
        labels=bp.get_xticklabels(),
        rotation=55,
        horizontalalignment='right',
        fontweight='light'
    )
    plt.show()


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #


def main(model: str = "all",
         verbose: bool = VERBOSE,
         load_full: bool = LOAD_FULL,
         exploration: bool = EXPLOATION,
         download_dataset: bool = DOWNLOAD_DATASET,
         tune_hyperparams: bool = HYPERPARAMS,
         get_feature_importances: bool = IMPORTANCES,
         ) -> None:
    """
    Main function, comprising the entire methodology of the program, acting as
    a high level overview of the full workflow.
    """
    start_time = time.time()

    # Download the dataset and load it into a Pandas DataFrame
    if download_dataset:
        print("Downloading the data...")
        get_covid_dataset()
        print("Data downloaded.\n")

    # For analysis' sake, naively load the full dataset if toggeled, and
    # visualise the proportion of missing values therein
    if load_full:
        print("Naively loading data in as a DataFrame...")
        naive_data = load_covid_dataset_naive()
        print("Data loaded!")
        print(f"Entire dataset size: {len(naive_data)}")
        missing_outcomes = percentage_missing(naive_data['outcome'])
        print(f"Proportion of missing outcome values: {missing_outcomes}.")
        print(naive_data.info())
        # display_nans(naive_data, "heatmap_1.png")
        del naive_data

    print("Loading in the data as a DataFrame, in chunks...")
    covid_data = load_covid_dataset_in_chunks()
    print("Data loaded successfully.\n")

    # display_nans(covid_data, "heatmap_2")
    covid_data = establish_data(covid_data, verbose)
    train_set, test_set = obtain_train_test_sets(covid_data)

    # Explote the training set
    if exploration:
        data_exploration(train_set)

    # Preprocess and split training data
    train_set = data_preparation(train_set)
    train_X, train_y = split_X_y(train_set)

    # Apply preprocessing metods and split test data **separately** from the
    # training data
    test_set = data_preparation(test_set)
    test_X, test_y = split_X_y(test_set)

    # Model 1: Logistic Regression
    if model == 'LR' or model == 'all':
        print("Commencing logistic regression.")

        # Obtain the logistic regression classifer prior to hyperparam. tuning
        lr_basic = logistic_regression(
            (train_X, train_y),
            get_basic_model=True
        )

        # Evaluate this basic logistic regression model
        analyse_results_text(
            'Basic Logistic Regression',
            lr_basic,
            (test_X, test_y),
            (train_X, train_y)
        )

        # Obtain the logistic regression model using tuned hyperparameters
        lr_tuned = logistic_regression(
            (train_X, train_y),
            get_best_params=tune_hyperparams
        )

        # Evaluate the tuned logistic regression classifier
        analyse_results_text(
            'Tuned Logistic Regression',
            lr_tuned,
            (test_X, test_y),
            (train_X, train_y)
        )
        confusion_matrix_heatmap(
            'Logistic Regression',
            lr_tuned,
            (test_X, test_y)
        )
        illustrate_roc_curves(
            (
                (lr_basic, "Basic Logistic Regression Classifier"),
                (lr_tuned, "Tuned Logistic Regression Classifier")
            ),
            (test_X, test_y),
            'ROC Curves for Comparing\n '
            'Basic and Tuned Logistic Regression Classifiers'
        )

        illustrate_pr_curves(
            (
                (lr_basic, "Basic Logistic Regression Classifier"),
                (lr_tuned, "Tuned Logistic Regression Classifier")
            ),
            (test_X, test_y),
            'Precision-Recall Curves for Comparing\n '
            'Basic and Tuned Logistic Regression Classifiers'
        )

        # Obtain the feature importances for the model
        if get_feature_importances:
            obtain_feature_importances(
                lr_tuned, (test_X, test_y),
                "Feature importance for Logistic Regression"
            )

    # Model 2: Random Forest Classifier
    if model == 'RFC' or model == 'all':
        print("Commencing random forest classifier.")

        # Obtain the basic variant of the Random Forest
        rfc_basic = random_forest_classifier(
            (train_X, train_y),
            get_basic_model=True
        )

        # Textually display the evaluation metrics for the basic variant
        analyse_results_text(
            'Basic Random Forest Classifier',
            rfc_basic,
            (test_X, test_y),
            (train_X, train_y)
        )

        # Obtain the RFC with tuned hyperparameters
        rfc_tuned = random_forest_classifier(
            (train_X, train_y),
            get_best_params=tune_hyperparams
        )

        # Texually analyse the results of the tuned RFC
        analyse_results_text(
            'Tuned Random Forest Classifier',
            rfc_tuned,
            (test_X, test_y),
            (train_X, train_y)
        )

        # Visually analyse the results of the tuned RFC
        confusion_matrix_heatmap(
            'Random Forest Classifier',
            rfc_tuned,
            (test_X, test_y)
        )
        illustrate_roc_curves(
            (
                (rfc_basic, "Basic Random Forest Classifier"),
                (rfc_tuned, "Tuned Random Forest Classifier")
            ),
            (test_X, test_y),
            'ROC Curves for Comparing\n '
            'Basic and Tuned Random Forest Classifiers'
        )

        illustrate_pr_curves(
            (
                (rfc_basic, "Basic Random Forest Classifier"),
                (rfc_tuned, "Tuned Random Forest Classifier")
            ),
            (test_X, test_y),
            'Precision-Recall Curves for Comparing\n '
            'Basic and Tuned Random Forest Classifiers'
        )

        # Use permutation importance to uncover the most valuable predictors
        if get_feature_importances:
            obtain_feature_importances(
                rfc_tuned, (test_X, test_y),
                "Feature importance for Random Forest Classification"
            )

    # Model 3: Support Vector Classifier
    if model == 'SVC' or model == 'all':
        print("Commencing support vector classifier.")

        # Obtain the generic SVC for the data
        svc_basic = support_vector_classifier(
            (train_X, train_y),
            get_basic_model=True
        )

        # Textually examine the generic SVC
        analyse_results_text(
            'Basic Support Vector Classifier',
            svc_basic,
            (test_X, test_y),
            (train_X, train_y)
        )

        # Obtain the SVC after hyperparameter tuning
        svc_tuned = support_vector_classifier(
            (train_X, train_y),
            get_best_params=tune_hyperparams
        )

        # Display the textual performance metrics for the tuned SVC
        analyse_results_text(
            'Tuned Support Vector Classifier',
            svc_tuned,
            (test_X, test_y),
            (train_X, train_y)
        )

        # Visualise the outcome of the tuned SVC
        confusion_matrix_heatmap(
            'Support Vector Classifier',
            svc_tuned,
            (test_X, test_y)
        )
        illustrate_roc_curves(
            (
                (svc_basic, "Basic Support Vector Classifier"),
                (svc_tuned, "Tuned Support Vector Classifier")
            ),
            (test_X, test_y),
            'ROC Curves for Comparing\n '
            'Basic and Tuned Support Vector Classifiers'
        )

        illustrate_pr_curves(
            (
                (svc_basic, "Basic Support Vector Classifier"),
                (svc_tuned, "Tuned Support Vector Classifier")
            ),
            (test_X, test_y),
            'Precision-Recall Curves for Comparing\n '
            'Basic and Tuned Support Vector Classifiers'
        )

        # Obtain the features most valuable for the tuned SVC
        if get_feature_importances:
            obtain_feature_importances(
                svc_tuned, (test_X, test_y),
                "Feature importance for Support Vector Classification"
            )

    # Directly compare the performances of all three models by plotting the
    # ROC curves and Precision-Recall curves for the tuned models on the same
    # canvas
    if model == 'all':
        illustrate_roc_curves(
            (
                (lr_tuned, "Logistic Regression Classifier Tuned"),
                (rfc_tuned, "Random Forest Classifier Tuned"),
                (svc_tuned, "Support Vector Classifier Tuned")
            ),
            (test_X, test_y),
            "ROC Curves for Comparing\nthe Tuned Models"
        )
        illustrate_pr_curves(
            (
                (lr_tuned, "Logistic Regression Classifier Tuned"),
                (rfc_tuned, "Random Forest Classifier Tuned"),
                (svc_tuned, "Support Vector Classifier Tuned")
            ),
            (test_X, test_y),
            'Precision-Recall Curves for Comparing\n '
            'the Tuned Models'
        )

    print("\nProgram finished.")
    print(f"Total elapsed time: {time.time() - start_time:.3f} seconds.")


if __name__ == '__main__':
    main()
