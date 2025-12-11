#!/usr/bin/env python
# coding: utf-8

# **Phase 2:**
# 
# Now you have had time to explore and get used to the dataset, use the data to answer some interesting questions. It is time to restart your analysis of the data in the attept to build a predictive model. Feel free to use what you learned in phase 1, for example, if you learned that some variables are highly correlated to the output that you want to predict, you want to make sure to keep them into your analysis. Please perform the following steps:
# 
# **Introduction: (5 pt)**
# 
# Formulate your research question so that you focus on predicting an outcome from the data (i.e prediction of Microcardial Infraction). It can be that your questions in Phase 1 have helped you to determine where to focus during phase 2, or you focus on all the data this time (not on a subset)
# 
# **Data preparation: (10 pt)**
# 
# Load the dataset in the same way as in Phase 1.
# Copy the link and load it as we do in our practicals. (5 pt)
# Select rows and columns relevant to your research question. (5 pt)
# 
# **Explore and clean the data by: (30 pt)**
# 
# Identify, report and correct issues with missing data (if any). (10 pt)
# Identify, report and correct issues with erroneous data (if any). (10 pt)
# Identify and correct outliers (if any). Explain your reasoning. (10 pt)
# 
# **Describe and visualize: (25 pt)**
# 
# Provide a summary of your cohort, this is a description of the final clean data. Make use of a table with descriptive statistics (i.e. means, medians, standard deviations) of important variables such as age, gender, outcome ect. Where possible, use visualisations. (10pt)
# Make the report interactive: Create at least one interactive visualisation using input from the user. (10pt)
# Turn your interactive report into an application using GitHub, Voila and Binder.** (5pt)
# 
# **Data analysis: (25 pt)**
# 
# Perform feature engineering on the data to better apply AI models to them (5pt)
# After splitting the data in a way that you can at a minimum test and train, apply several prediction models to your data. (10pt)
# Use performance metrics to determine the best model (5 pt)
# Apply further hyper-parameter tuning, or cross validation if possible. (5 pt)
# 
# **Conclusion: (5 pt)**
# 
# Summarize the work and the main findings related to the initial research question. (5 pt)

# RQ2 - In adults in the Framingham Heart Study (P), how well do baseline cardiovascular risk factors, including sex (I), predict incident cardiovascular events and all-cause mortality over 24 years (O, T), and does predictive performance or estimated risk differ between men and women (C)?
# - Outcomes measured: Myocardial infarction, CVD, Stroke, Death
# 
# 

# In[68]:


import pandas as pd
import numpy as np


# In[69]:


# First import data file
CVD = pd.read_csv('https://raw.githubusercontent.com/LUCE-Blockchain/Databases-for-teaching/refs/heads/main/Framingham%20Dataset.csv')
CVD.head()


# Key outcomes: Myocardial infarction, Stroke, CVD, Death
# 
# Aims:
# 1. describe gender differences in risk factors
# 2. build predictive models for selected outcomes

# # Subset

# In[70]:


unique_periods = CVD['PERIOD'].unique()
print(unique_periods)


# In[71]:


cvd_by_period = {}
for period in unique_periods:
    cvd_by_period[period] = CVD[CVD['PERIOD'] == period]

print("DataFrames split by 'PERIOD' column:")
for period, df in cvd_by_period.items():
    print(f"nPeriod {period} DataFrame (first 5 rows):")
    print(df.head())


# #Select period 1 only

# In[72]:


cvd_by_period[1].describe()


# In[73]:


cvd_by_period[1].info()


# # Create risk profile df

# In[74]:


selected_columns = [
    'AGE',
    'SEX',
    'CURSMOKE',
    'CIGPDAY',
    'BMI',
    'SYSBP',
    'DIABP',
    'HEARTRTE',
    'TOTCHOL',
    'DIABETES',
    'PREVHYP',
    'PREVCHD',
    'PREVAP',
    'PREVMI',
    'PREVSTRK',
    'BPMEDS'
]

risk_profile_df = cvd_by_period[1][selected_columns]
risk_profile_df = risk_profile_df.groupby('SEX')

print("Risk Profile grouped by Sex for Period 1:")
print(risk_profile_df.head().sort_values(by='SEX'))
print(risk_profile_df.describe())


# In[75]:


risk_profile_df.head()


# ### Reshaping Descriptive Statistics for Risk Profile
# - AI generated code

# In[76]:


# Get the descriptive statistics, which is already grouped by 'SEX'
desc_stats = risk_profile_df.describe()

# Transpose the DataFrame
# This will make the original multi-index columns (e.g., (AGE, mean)) into rows
# and the original index (SEX) into columns.
desc_stats_transposed = desc_stats.T

# Rename the columns for better readability (SEX 1 and SEX 2)
desc_stats_transposed.columns = [f'SEX_{col}' for col in desc_stats_transposed.columns]

# Reset the index to turn the multi-index (variable, statistic) into regular columns
desc_stats_final = desc_stats_transposed.reset_index()

# Rename the index levels for clarity
desc_stats_final.rename(columns={'level_0': 'Variable', 'level_1': 'Statistic'}, inplace=True)

print("Descriptive statistics reshaped (more rows, fewer columns):")
display(desc_stats_final)


# ##Check consistency

# In[77]:


import pandas as pd

def check_smoking_consistency(df):
    """
    Checks for inconsistencies where CURSMOKE is 0 but CIGPDAY is > 0.

    Args:
        df (pd.DataFrame): The input DataFrame containing 'CURSMOKE' and 'CIGPDAY' columns.

    Returns:
        pd.DataFrame: A DataFrame containing rows with inconsistencies.
    """

    inconsistent_data = df[(df['CURSMOKE'] == 0) & (df['CIGPDAY'] > 0)]

    return inconsistent_data

inconsistent_smoking_data = check_smoking_consistency(cvd_by_period[1])

if not inconsistent_smoking_data.empty:
    print("Found inconsistencies in smoking data (CURSMOKE=0 but CIGPDAY>0):")
    display(inconsistent_smoking_data[['CURSMOKE', 'CIGPDAY']].head())
    print(f"Total inconsistent records: {len(inconsistent_smoking_data)}")
else:
    print("No inconsistencies found where CURSMOKE=0 and CIGPDAY>0 in the data.")


# #Create outcomes df
# 

# In[78]:


# outcomes_df
selected_outcomes_columns = [
    'ANGINA',
    'HOSPMI',
    'STROKE',
    'ANYCHD',
    'HYPERTEN',
    'DEATH',
    'SEX'
]

# Create outcomes_df from cvd_by_period[1]
outcomes_df = cvd_by_period[1][selected_outcomes_columns]

# Group by SEX and display descriptive statistics
print("Outcomes Profile grouped by Sex for Period 1:")
print(outcomes_df.groupby('SEX').describe())


# ###Check event counts

# In[79]:


outcome_columns = [
    'ANGINA',
    'HOSPMI',
    'STROKE',
    'ANYCHD',
    'HYPERTEN',
    'DEATH']

for period, df in cvd_by_period.items():
    print(f"Outcome Counts for Period {period}:")
    for col in outcome_columns:
        if col in df.columns:
            # Calculate the number of events (sum of 1s) and total observations
            event_count = df[col].sum()
            total_observations = df[col].count()

            if total_observations > 0:
                event_percentage = (event_count / total_observations) * 100
                print(f"  {col}: {int(event_count)} events out of {total_observations} observations ({event_percentage:.2f}%)")
            else:
                print(f"  {col}: No observations.")
        else:
            print(f"  {col}: Column not found in Period {period} DataFrame.")


# #Missingness

# In[80]:


# count
risk_profile_df = cvd_by_period[1][selected_columns]
missing_counts = risk_profile_df.isnull().sum()

# percentage
missing_percentages = (risk_profile_df.isnull().sum() / len(risk_profile_df)) * 100

missing_info = pd.DataFrame({
    'Missing Count': missing_counts,
    'Missing Percentage': missing_percentages
})

missing_info = missing_info[missing_info['Missing Count'] > 0].sort_values(by='Missing Percentage', ascending=False)

print("Missing Values Information for Risk Profile DataFrame:")
display(missing_info)


# In[81]:


import matplotlib.pyplot as plt
import seaborn as sns

if not missing_info.empty:
    plt.figure(figsize=(10, 6))
    sns.barplot(x=missing_info.index, y='Missing Percentage', data=missing_info, palette='hls', hue = missing_info.index, legend = False)
    plt.title('Percentage of Missing Values per Column in Risk Profile DataFrame')
    plt.xlabel('Columns')
    plt.ylabel('Missing Percentage (%)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
else:
    print("No missing values to display in the risk profile dataframe.")


# In[82]:


# count
missing_counts_outcomes = outcomes_df.isnull().sum()

# percentage
missing_percentages_outcomes = (outcomes_df.isnull().sum() / len(outcomes_df)) * 100

missing_info_outcomes = pd.DataFrame({
    'Missing Count': missing_counts_outcomes,
    'Missing Percentage': missing_percentages_outcomes
})

missing_info_outcomes = missing_info_outcomes[missing_info_outcomes['Missing Count'] > 0].sort_values(by='Missing Percentage', ascending=False)

print("Missing Values Information for Outcomes DataFrame:")
display(missing_info_outcomes)


# ## Handling missing values
# - variables with missings had <5%, so we drop them

# In[49]:


columns_for_dropna_check = selected_columns # These are the columns to check for NaNs
columns_to_extract = selected_columns + ['RANDID'] # These are the columns we need from cvd_by_period[1]

temp_df_for_dropna = cvd_by_period[1][columns_to_extract].copy()

initial_rows = len(temp_df_for_dropna)
print(f"Initial number of rows: {initial_rows}")

dropped_df = temp_df_for_dropna.dropna(subset=columns_for_dropna_check)

final_rows = len(dropped_df)
rows_dropped = initial_rows - final_rows

print(f"Number of rows after dropping missing values: {final_rows}")
print(f"Total rows dropped: {rows_dropped}")

kept_randids = dropped_df['RANDID'].unique()

cvd_by_period[1] = cvd_by_period[1][cvd_by_period[1]['RANDID'].isin(kept_randids)].copy()

print("\nUpdated 'cvd_by_period[1]' DataFrame after dropping rows with missing values in risk profile columns.")
print(f"New number of rows in cvd_by_period[1]: {len(cvd_by_period[1])}")

# Verify that there are no more missing values in the selected risk profile columns within cvd_by_period[1]
print("\nMissing values in selected risk profile columns after dropping (should be all 0s):")
display(cvd_by_period[1][selected_columns].isnull().sum())


# # Inspecting variables for distribution & outliers

# In[52]:


import matplotlib.pyplot as plt
import seaborn as sns

print("Plotting distributions for selected risk profile columns:")

numerical_cols_for_dist = [
    'AGE',
    'TOTCHOL',
    'SYSBP',
    'DIABP',
    'CIGPDAY',
    'BMI',
    'HEARTRTE'
]

for col in numerical_cols_for_dist:
    plt.figure(figsize=(10, 5))
    # Histogram:
    sns.histplot(data=risk_profile_df, x=col, kde=True, bins=30)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

    # Boxplots
    plt.figure(figsize=(10, 2))
    sns.boxplot(data=risk_profile_df, x=col)
    plt.title(f'Box Plot of {col}')
    plt.xlabel(col)
    plt.tight_layout()
    plt.show()


# NB: CIGPDAY: Spike @ 20 - Brainstorm reasons + document

# Physiological limits:
# - Age: 18-110
# - Cigpday: <80
# - BMI: 10-70
# - SYSBP: 60-300
# - DIASBP: 30-150
# - HR: 30 - (220- 32)
# - TOTCHOL: 70–600

# ### Applying Physiological Limits (Winsorization)

# In[51]:


df_period1 = cvd_by_period[1].copy() # Work on a copy

physiological_limits = {
    'AGE': {'min': 18, 'max': 110},
    'CIGPDAY': {'min': 0, 'max': 80},
    'BMI': {'min': 10, 'max': 70},
    'SYSBP': {'min': 60, 'max': 300},
    'DIABP': {'min': 30, 'max': 150},
    'HEARTRTE': {'min': 30, 'max': 188}, # Calculated as 220 - 32
    'TOTCHOL': {'min': 70, 'max': 600}
}

for col, limits in physiological_limits.items():
    if col in df_period1.columns:
        original_min = df_period1[col].min()
        original_max = df_period1[col].max()

        # Winsorize lower bound
        df_period1[col] = df_period1[col].clip(lower=limits['min'])
        # Winsorize upper bound
        df_period1[col] = df_period1[col].clip(upper=limits['max'])

        winsorized_min = df_period1[col].min()
        winsorized_max = df_period1[col].max()

        print(f"  Column '{col}':")
        print(f"    Original Min/Max: ({original_min:.2f}, {original_max:.2f})")
        print(f"    Applied Limits: ({limits['min']}, {limits['max']})")
        print(f"    Winsorized Min/Max: ({winsorized_min:.2f}, {winsorized_max:.2f})")

cvd_by_period[1] = df_period1

print("\nDescriptive statistics for affected columns after winsorization:")
display(cvd_by_period[1][list(physiological_limits.keys())].describe())


# In[ ]:





# Next steps: boxplots,
# missings,
# exclusion strategy,

# 
#  Filter observations:
# For incident outcomes, exclude those with prevalent disease (e.g. PREVMI=1 if predicting new MI).
# Optionally limit to first exam (PERIOD = 1) for consistent baseline.
# 
# 3. Data Cleaning and Quality Control (30 pts)
# 
# Missing data
#  Identify missing values per variable.
# Decide on strategy:
# Impute (mean/median for continuous, mode for categorical) or
# Drop if small proportion.
# -  Document which variables were imputed and how.
# 
# Erroneous data:
# Check variable ranges against data dictionary (e.g. SYSBP 83.5–295).
# Flag impossible or implausible values (e.g. negative, zero for BMI).
# Correct or remove those rows; explain reasoning.
# 
# Outliers:
# Plot distributions (boxplots/histograms) for key continuous variables.
# Apply IQR rule or z-score threshold (>3 SD) to flag extreme outliers.
#  Decide whether to cap (winsorize) or remove.
#  Explain justification in notebook text.
# 

# In[ ]:





# # Task
# Create six separate DataFrames for specific incident outcomes from, excluding participants with prevalent conditions for each outcome, and store them in a dictionary called incident_outcome_dfs. Each DataFrame will include RANDID, selected risk factors, and the respective incident outcome variable. The outcomes to create are: incident Myocardial Infarction or Fatal CHD, incident Stroke, incident Any CHD, incident CVD, incident Angina, and all-cause Mortality.
