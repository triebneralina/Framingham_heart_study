import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -------------------------------------------------------------------
# CONFIGURE
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Framingham CVD Analysis",
    layout="wide",
)

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "LUCE-Blockchain/Databases-for-teaching/refs/heads/main/"
    "Framingham%20Dataset.csv"
)

# -------------------------------------------------------------------
# DATA LOADING & BASIC PREP
# -------------------------------------------------------------------
## Cache_data = reuse the stored result, does not re-run function = prevents slow computing
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL)
    return df


@st.cache_data
def get_period_1_df(df):
    if "PERIOD" not in df.columns:
        return df.copy()
    return df[df["PERIOD"] == 1].copy()

df = load_data()
period1_df = get_period_1_df(df)


# Columns used
RISK_PROFILE_COLUMNS = [
    "AGE",
    "SEX",
    "CURSMOKE",
    "CIGPDAY",
    "BMI",
    "SYSBP",
    "DIABP",
    "HEARTRTE",
    "TOTCHOL",
    "DIABETES",
    "PREVHYP",
    "PREVCHD",
    "PREVAP",
    "PREVMI",
    "PREVSTRK",
    "BPMEDS",
]

# For plotting distributions and boxplots
CONTINUOUS_COLUMNS_ONLY = [
    "AGE",
    "CIGPDAY",
    "BMI",
    "SYSBP",
    "DIABP",
    "HEARTRTE",
    "TOTCHOL",
]

BINARY_COLUMNS_ONLY = [    
    "CURSMOKE",
    "DIABETES",
    "PREVHYP",
    "PREVCHD",
    "PREVAP",
    "PREVMI",
    "PREVSTRK",
    "BPMEDS",]

OUTCOME_COLUMNS = [
    "CVD",
    "DEATH",
]

OUTCOME_COLUMNS_FOR_COUNTS = [
    "CVD",
    "DEATH",
]


PHYSIOLOGICAL_LIMITS = {
    "AGE": {"min": 18, "max": 110},
    "CIGPDAY": {"min": 0, "max": 80},
    "BMI": {"min": 10, "max": 70},
    "SYSBP": {"min": 60, "max": 300},
    "DIABP": {"min": 30, "max": 150},
    "HEARTRTE": {"min": 30, "max": 188},
    "TOTCHOL": {"min": 70, "max": 600},
}


def make_risk_profile_df(period1_df):
    """
    Generic risk profile (baseline) dataframe:
    just the risk factor columns, no outcomes.
    Useful for descriptive stats by sex, etc.
    """
    cols = [c for c in RISK_PROFILE_COLUMNS if c in period1_df.columns]
    return period1_df[cols].copy()

def make_outcomes_df(period1_df):
    """ Outcomes dataframe: SEX + CVD/ DEATH (If present)."""
    cols = ["SEX"] + [c for c in OUTCOME_COLUMNS if c in period1_df.columns]
    cols = [c for c in cols if c in period1_df.columns]
    return period1_df[cols].copy()


# --- Outcome-specific datasets --- #

# Define outcome-specific predictor sets for CVD and death separately
CVD_PREDICTOR_COLUMNS = [
    "AGE",
    "SEX",
    "CURSMOKE",
    "CIGPDAY",
    "BMI",
    "SYSBP",
    "DIABP",
    "HEARTRTE",
    "TOTCHOL",
    "DIABETES",
    "PREVHYP",
    "BPMEDS",
]

DEATH_PREDICTOR_COLUMNS = [
    "AGE",
    "SEX",
    "CURSMOKE",
    "CIGPDAY",
    "BMI",
    "SYSBP",
    "DIABP",
    "HEARTRTE",
    "TOTCHOL",
    "DIABETES",
    "PREVHYP",
    "PREVCHD",
    "PREVAP",
    "PREVMI",
    "PREVSTRK",
    "BPMEDS",
]


def make_cvd_dataset(period1_df):
    """
    Build the analytic dataset for incident CVD:
    - Restrict to Period 1 (baseline)
    - Exclude participants with prevalent CHD or stroke at baseline
    - Return predictors + CVD outcome (ready for train/test split)
    """
    df = period1_df.copy()

    # 1) Exclude prevalent CVD (CHD or stroke) at baseline
    #    Only apply if those columns exist
    if "PREVCHD" in df.columns:
        df = df[df["PREVCHD"] == 0]
    if "PREVSTRK" in df.columns:
        df = df[df["PREVSTRK"] == 0]

    # 2) Keep predictors + outcome
    cols = [c for c in CVD_PREDICTOR_COLUMNS if c in df.columns]
    if "CVD" in df.columns:
        cols = cols + ["CVD"]
    else:
        raise KeyError("CVD outcome column not found in dataframe.")

    return df[cols].copy()


def make_death_dataset(period1_df):
    """
    Build the analytic dataset for all-cause mortality:
    - Restrict to Period 1 (baseline)
    - Keep everyone (no exclusion on previous disease)
    - Return predictors + DEATH outcome (ready for train/test split)
    """
    df = period1_df.copy()

    # 1) Keep predictors + outcome
    cols = [c for c in DEATH_PREDICTOR_COLUMNS if c in df.columns]
    if "DEATH" in df.columns:
        cols = cols + ["DEATH"]
    else:
        raise KeyError("DEATH outcome column not found in dataframe.")

    return df[cols].copy()


# -------------------------------------------------------------------
# CHECKS & CLEANING FUNCTIONS
# -------------------------------------------------------------------
def check_smoking_consistency(period1_df):
    """
    Checks for inconsistencies where CURSMOKE is 0 but CIGPDAY is > 0
    """
    if "CURSMOKE" not in period1_df.columns or "CIGPDAY" not in period1_df.columns:
        return pd.DataFrame()
    inconsistent_data = period1_df[(period1_df["CURSMOKE"] == 0) & (period1_df["CIGPDAY"] > 0)]
    return inconsistent_data


def compute_missing_info(df):
    missing_counts = df.isnull().sum()
    missing_percentages = df.isnull().mean() * 100
    missing_info = pd.DataFrame(
        {
            "Missing Count": missing_counts,
            "Missing Percentage": missing_percentages,
        }
    )
    missing_info = missing_info[missing_info["Missing Count"] > 0]
    missing_info = missing_info.sort_values(
        by="Missing Percentage", ascending=False
    )
    return missing_info


def apply_dropna_on_risk_profile(period1_df):
    """
    Handles missingeness in risk profile:
      - extract RANDID + risk profile cols
      - drop rows with NaNs in risk profile
      - keep only rows with kept RANDIDs
    """
    if "RANDID" not in period1_df.columns:
        # Fallback: just dropna on risk profile columns directly
        cols = [c for c in RISK_PROFILE_COLUMNS if c in period1_df.columns]
        return period1_df.dropna(subset=cols)

    cols_for_check = [c for c in RISK_PROFILE_COLUMNS if c in period1_df.columns]
    cols_to_extract = cols_for_check + ["RANDID"]

    temp_df = period1_df[cols_to_extract].copy()
    dropped_df = temp_df.dropna(subset=cols_for_check)

    kept_randids = dropped_df["RANDID"].unique()
    filtered = period1_df[period1_df["RANDID"].isin(kept_randids)].copy()
    return filtered


def winsorize_period1(period1_df):
    """
    Apply clip() according to PHYSIOLOGICAL_LIMITS.
    """
    df = period1_df.copy()
    for col, limits in PHYSIOLOGICAL_LIMITS.items():
        if col in df.columns:
            df[col] = df[col].clip(lower=limits["min"], upper=limits["max"])
    return df


# -------------------------------------------------------------------
# PLOTTING HELPERS: NEEDS REVIEW!!!!!!
# -------------------------------------------------------------------
def plot_missing_bar(missing_info, title):
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(
        x=missing_info.index,
        y="Missing Percentage",
        data=missing_info,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Columns")
    ax.set_ylabel("Missing Percentage (%)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig


SEX_COLORS = {1: "paleturquoise", 2: "pink"}
def plot_hist_by_sex(df, col):
    """
    Histogram of a numeric column grouped by SEX using dodge-style bars.
    Colors: paleturquoise (SEX=0), pink (SEX=1)
    """

    fig, ax = plt.subplots(figsize=(7, 4))

    sns.histplot(
        data=df,
        x=col,
        hue="SEX",
        multiple="dodge",                     # side-by-side bars
        palette=[SEX_COLORS.get(1), SEX_COLORS.get(2)],
        alpha=1.0,
        kde=True,
        ax=ax
    )

    ax.set_title(f"{col} Distribution by Sex", fontsize=14, fontweight="bold")
    ax.set_xlabel(col)
    ax.set_ylabel("Count")

    fig.tight_layout()
    return fig

def plot_box_by_sex(df, col):
    """
    Boxplot of a numeric column grouped by SEX.
    """

    fig, ax = plt.subplots(figsize=(6, 3))

    sns.boxplot(
        data=df,
        x="SEX",
        y=col,
        palette=[SEX_COLORS.get(1), SEX_COLORS.get(2)],
        ax=ax
    )

    ax.set_title(f"Box Plot of {col} by Sex", fontsize=12, fontweight="bold")
    ax.set_xlabel("SEX")
    ax.set_ylabel(col)

    fig.tight_layout()
    return fig

def plot_distribution_by_sex(df, col):
    """
    Returns both the histogram and boxplot for a column grouped by SEX.
    """
    fig_hist = plot_hist_by_sex(df, col)
    fig_box = plot_box_by_sex(df, col)
    return fig_hist, fig_box

def plot_binary_presence_percent_by_sex(df, binary_col, sex_col="SEX"):
    """
    Plot percentage of individuals with binary_col == 1 within each sex.
    One bar per sex:
        - Male (paleturquoise)
        - Female (pink)
    """

    plt.style.use("ggplot")

    # Keep only sex + variable
    data = df[[sex_col, binary_col]].dropna().copy()

    # Compute the percentage of "1" per sex
    pct = (
        data.groupby(sex_col)[binary_col]
        .apply(lambda x: (x == 1).mean() * 100)
        .reindex([1, 2])      # Ensure order: Male, Female
    )

    # Map sex codes → labels
    sex_label_map = {1: "Male", 2: "Female"}
    labels = [sex_label_map.get(s, str(s)) for s in pct.index]

    # Colors by sex
    colors = ["paleturquoise", "pink"]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, pct.values, color=colors)

    # Label percentages on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2,
            height,
            f"{height:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_title(f"{binary_col} (value = 1) by Sex", fontsize=14, fontweight="bold")
    ax.set_xlabel("Sex")
    ax.set_ylabel("Percentage (%)")

    fig.tight_layout()
    return fig

#--------------------------------------------------------------------
# DATA PREPARATION
#--------------------------------------------------------------------
def prepare_model_data(df, outcome_col):
    """
    Helper to:
    - Drop rows with missing values
    - Split predictors vs outcome
    - Keep only numeric predictors for modeling
    """
    df_clean = df.dropna().copy()

    if outcome_col not in df_clean.columns:
        raise KeyError(f"Outcome column {outcome_col} not found in dataframe.")

    y = df_clean[outcome_col].astype(int)
    X = df_clean.drop(columns=[outcome_col])

    # keep numeric predictors only
    X = X.select_dtypes(include=[np.number])

    return X, y


def train_test_and_scale(X, y, test_size=0.3, random_state=42):
    """
    Train/test split + standardization of numeric predictors.
    Returns:
      X_train, X_test, y_train, y_test, X_train_scaled_df, X_test_scaled_df, scaler
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Wrap back into DataFrames for easier viewing
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

    return X_train, X_test, y_train, y_test, X_train_scaled_df, X_test_scaled_df, scaler


def plot_corr_heatmap(df, title):
    """
    Correlation heatmap for numeric columns in df.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        corr,
        cmap="coolwarm",
        center=0,
        square=True,
        annot=False,
        ax=ax,
    )
    ax.set_title(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


# -------------------------------------------------------------------
# MAIN APP
# -------------------------------------------------------------------
def main():
    st.title("Framingham Heart Study: CVD Risk Analysis by Sex")
    st.write("Interactive report on the loading, cleaning and analytic datasets")

    # Load data
    df = load_data()
    period1_df = get_period_1_df(df)

    # Analytic Datasets
    incident_cvd_df = make_cvd_dataset(period1_df)
    incident_death_df = make_death_dataset(period1_df)

    # Sidebar controls
    st.sidebar.header("Controls")

    view = st.sidebar.radio(
        "View",
        [
            "Raw data overview",
            "Risk profile by sex",
            "Outcomes by sex",
            "Outcome event rates",
            "Consistency checks",
            "Missing data",
            "Winsorization summary",
            "Modeling: CVD",
            "Modeling: DEATH",
        ],
    )

    # ----------------------------------------------------------------
    # RAW DATA OVERVIEW
    # ----------------------------------------------------------------
    if view == "Raw data overview":
        st.subheader("Raw Data: Period 1")

        st.write("**Header: Period 1**")
        st.dataframe(period1_df.head(10))

        st.write("**Basic descriptive statistics (numeric columns):**")
        st.dataframe(period1_df.describe())

        st.write("**Columns & dtypes:**")
        st.dataframe(
            pd.DataFrame(
                {
                    "Column": period1_df.columns,
                    "Dtype": period1_df.dtypes.astype(str).values,
                }
            )
        )

    # ----------------------------------------------------------------
    # RISK PROFILE BY SEX: add interactive component to shorten table 
    # ----------------------------------------------------------------
    elif view == "Risk profile by sex":
        st.subheader(f"Risk Profile by Sex: Period 1")

        risk_df = make_risk_profile_df(period1_df)

        st.write("**Selected risk profile columns:**")
        st.write(RISK_PROFILE_COLUMNS)

        if risk_df.empty:
            st.warning("No risk profile columns found for this period.")
        else:
            grouped = risk_df.groupby("SEX")
            st.write("**Grouped by SEX - first rows from each group:**")
            st.dataframe(grouped.head().sort_values(by="SEX"))

            st.write("**Descriptive statistics by SEX:**")
            st.dataframe(grouped.describe())

            # Reshape like notebook (describe -> transpose, etc.)
            desc_stats = grouped.describe()
            desc_stats_transposed = desc_stats.T
            desc_stats_transposed.columns = [
                f"SEX_{col}" for col in desc_stats_transposed.columns
            ]
            desc_stats_final = desc_stats_transposed.reset_index()
            desc_stats_final.rename(
                columns={"level_0": "Variable", "level_1": "Statistic"},
                inplace=True,
            )

            st.write("**Reshaped descriptive statistics (rows = variable/statistic):**")
            st.dataframe(desc_stats_final)

        #-------------------Continuous distributions---------------------------------------    
        st.subheader(f"Distributions of Continuous Risk Profile Variables")

        risk_df = make_risk_profile_df(period1_df)
        numeric_cols = [
            c for c in CONTINUOUS_COLUMNS_ONLY
            if c in risk_df.columns and pd.api.types.is_numeric_dtype(risk_df[c])
        ]

        if not numeric_cols:
            st.warning("No numeric risk profile columns found for this period.")
        else:
            col_choice = st.selectbox("Select a continuous variable to plot", numeric_cols)
            fig_hist, fig_box = plot_distribution_by_sex(risk_df, col_choice)

            st.pyplot(fig_hist)
            st.pyplot(fig_box)

        #---------------------------Binary Distributions----------------------------------------------
        ##FIX LEGEND (0/1) COLOURS
        st.subheader(f"Distributions of Binary Risk Profile Variables")
        
        risk_df = make_risk_profile_df(period1_df)
        binary_cols = [
            c for c in BINARY_COLUMNS_ONLY
            if c in risk_df.columns and pd.api.types.is_numeric_dtype(risk_df[c])
        ]        
        if not binary_cols:
            st.warning("No binary variables found in the risk profile.")
        else:
            bin_choice = st.selectbox(
                "Select binary variable to plot",
                binary_cols,
            )

            fig_bar = plot_binary_presence_percent_by_sex(
                period1_df,
                binary_col=bin_choice,
                sex_col="SEX",
            )
            st.pyplot(fig_bar)
    # ----------------------------------------------------------------
    # OUTCOMES BY SEX: ADD BAR PLOTS !!!!!
    # ----------------------------------------------------------------
    elif view == "Outcomes by sex":
        st.subheader(f"Outcomes Profile by Sex: Period 1")

        outcomes_df = make_outcomes_df(period1_df)

        st.write("**Outcome columns used:**")
        st.write(OUTCOME_COLUMNS)

        if outcomes_df.empty:
            st.warning("No outcome columns found.")
        else:
            grouped = outcomes_df.groupby("SEX")
            st.write("**Descriptive statistics by SEX:**")
            st.dataframe(grouped.describe())

    # ----------------------------------------------------------------
    # OUTCOME EVENT RATES FOR ALL PERIODS
    # ----------------------------------------------------------------
    elif view == "Outcome event rates":
        st.subheader("Outcome Event Counts and Percentages")

        rows = []
        for col in OUTCOME_COLUMNS_FOR_COUNTS:
            if col in period1_df.columns:
                event_count = period1_df[col].sum()
                total_obs = period1_df[col].count()
                if total_obs > 0:
                    event_pct = (event_count / total_obs) * 100
                else:
                    event_pct = np.nan
                rows.append(
                     {
                        "Outcome": col,
                        "Events": int(event_count),
                        "Total": int(total_obs),
                        "Event %": event_pct,
                    }
                )
            else:
                rows.append(
                    {
                        "Outcome": col,
                        "Events": np.nan,
                        "Total": np.nan,
                        "Event %": np.nan,
                    }
                )
        st.dataframe(pd.DataFrame(rows))

    # ----------------------------------------------------------------
    # CONSISTENCY
    # ----------------------------------------------------------------
    elif view == "Consistency checks":
        st.subheader(f"Smoking Data Consistency")
        inconsistent_data = check_smoking_consistency(period1_df)

        st.markdown(
            "- **CURSMOKE**: current smoking status (0 = no, 1 = yes)  \n"
            "- **CIGPDAY**: cigarettes per day"
        )

        if inconsistent_data.empty:
            st.success("No inconsistencies found (CURSMOKE=0 & CIGPDAY>0).")
        else:
            st.error(
                f"Found {len(inconsistent_data)} inconsistent rows "
                "(CURSMOKE=0 but CIGPDAY>0)."
            )
            st.dataframe(inconsistent_data[["CURSMOKE", "CIGPDAY"]].head(20))
        
        #----------------PREV Consistency----------------------------



        

    # ----------------------------------------------------------------
    # MISSING DATA
    # ----------------------------------------------------------------
    elif view == "Missing data":
        st.subheader("Missing data overview: Period 1")

        #-------------- 1. Risk Profile-------------------------------
        st.markdown(f"Missing Data: Risk Profile for Period 1")

        risk_df = make_risk_profile_df(period1_df)
        missing_risk = compute_missing_info(risk_df)

        if missing_risk.empty:
            st.success("No missing values in selected risk profile columns.")
        else:
            st.write("**Missing values table:**")
            st.dataframe(missing_risk)

            fig = plot_missing_bar(
                missing_risk,
                "Percentage of Missing Values per Column in Risk Profile",
            )
            st.pyplot(fig)

        st.markdown("---")

        #-----------------2. Outcome variable --------------------------
        st.markdown(f"Missing Data: Outcomes - Period 1")

        outcomes_df = make_outcomes_df(period1_df)
        missing_outcomes = compute_missing_info(outcomes_df)

        if missing_outcomes.empty:
            st.success("No missing values in selected outcome columns.")
        else:
            st.write("**Missing values table:**")
            st.dataframe(missing_outcomes)

            fig = plot_missing_bar(
                missing_outcomes,
                "Percentage of Missing Values per Column in Outcomes",
            )
            st.pyplot(fig)

        st.markdown("---")   

        #------------------3. Modeling datasets -----------------------------------------------
        st.markdown("Missing data in the Analytic Modeling datasets (i.e. CVD and DEATH datasets)")

        dataframes_to_check = {
            "Incident CVD Dataframe": incident_cvd_df,
            "Incident Death Dataframe": incident_death_df,
        }

        for name, df_model in dataframes_to_check.items():
            st.markdown(f"### {name}")
            missing_info_df = compute_missing_info(df_model)

            if missing_info_df.empty:
                st.success("No missing values found.")
            else:
                st.write("Missing values table:")
                st.dataframe(missing_info_df)

                fig = plot_missing_bar(missing_info_df, f"Percentage of missing values - {name}")
                st.pyplot(fig)

    # ----------------------------------------------------------------
    # WINSORIZATION SUMMARY
    # ----------------------------------------------------------------
    elif view == "Winsorization summary":
        st.subheader("Winsorization Summary")

        # Start from original Period 1 (not yet cleaned) for demonstration
        raw_period1 = period1_df.copy()

        # Step 1: Drop rows with missing risk profile values
        dropped_df = apply_dropna_on_risk_profile(raw_period1)

        st.write(
            f"Rows before dropping missing risk profile values: "
            f"{len(raw_period1)}"
        )
        st.write(f"Rows after dropping: {len(dropped_df)}")

        # Step 2: Winsorize physiological variables
        before_stats = dropped_df[list(PHYSIOLOGICAL_LIMITS.keys())].describe()

        winsorized_df = winsorize_period1(dropped_df)
        after_stats = winsorized_df[list(PHYSIOLOGICAL_LIMITS.keys())].describe()

        st.markdown("### Descriptive Statistics Before Winsorization")
        st.dataframe(before_stats)

        st.markdown("### Descriptive Statistics After Winsorization")
        st.dataframe(after_stats)

        # Min/max comparison
        st.markdown("### Min/Max Comparison (Before vs After)")
        summary_rows = []
        for col, limits in PHYSIOLOGICAL_LIMITS.items():
            if col in winsorized_df.columns:
                summary_rows.append(
                    {
                        "Variable": col,
                        "Original Min": before_stats.loc["min", col],
                        "Original Max": before_stats.loc["max", col],
                        "Applied Min Limit": limits["min"],
                        "Applied Max Limit": limits["max"],
                        "Winsorized Min": after_stats.loc["min", col],
                        "Winsorized Max": after_stats.loc["max", col],
                    }
                )
        if summary_rows:
            st.dataframe(pd.DataFrame(summary_rows))

    # ----------------------------------------------------------------
    # MODELING: CVD
    # ----------------------------------------------------------------
    elif view == "Modeling: CVD":
        st.subheader("Modeling: Incident CVD (Period 1)")

        st.markdown("### 1. Analytic dataset overview")
        st.write("Shape of incident CVD dataset (rows, columns):", incident_cvd_df.shape)
        st.write("Columns:")
        st.write(list(incident_cvd_df.columns))

        st.write("Class balance (CVD = 0/1):")
        st.dataframe(incident_cvd_df["CVD"].value_counts().rename("Count").to_frame())

        st.markdown("### 2. Train/test split and standardization")

        # Prepare X, y
        X_cvd, y_cvd = prepare_model_data(incident_cvd_df, outcome_col="CVD")

        st.write("Predictor matrix shape:", X_cvd.shape)
        st.write("Outcome vector length:", len(y_cvd))

        # Train/test split + scaling
        (
            X_train,
            X_test,
            y_train,
            y_test,
            X_train_scaled,
            X_test_scaled,
            scaler_cvd,
        ) = train_test_and_scale(X_cvd, y_cvd, test_size=0.3, random_state=42)

        st.write("Training set size:", X_train.shape[0])
        st.write("Test set size:", X_test.shape[0])

        with st.expander("Show first 5 rows of original vs standardized predictors (train set)"):
            st.write("Original X_train (head):")
            st.dataframe(X_train.head())
            st.write("Standardized X_train (head):")
            st.dataframe(X_train_scaled.head())

        st.markdown("### 3. Correlation heatmap of predictors")
        fig_corr = plot_corr_heatmap(X_cvd, "Correlation Heatmap – CVD Predictors")
        st.pyplot(fig_corr)

        st.markdown("### 4. Logistic regression model – CVD")

        # Fit logistic regression on standardized predictors
        model_cvd = LogisticRegression(solver="liblinear")
        model_cvd.fit(X_train_scaled, y_train)

        # Predictions
        y_train_pred = model_cvd.predict(X_train_scaled)
        y_test_pred = model_cvd.predict(X_test_scaled)

        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)

        st.write(f"Training accuracy: **{train_acc:.3f}**")
        st.write(f"Test accuracy: **{test_acc:.3f}**")

        # Confusion matrix on test set
        cm = confusion_matrix(y_test, y_test_pred)
        cm_df = pd.DataFrame(
            cm,
            index=["Actual 0", "Actual 1"],
            columns=["Pred 0", "Pred 1"],
        )
        st.markdown("**Confusion matrix (test set):**")
        st.dataframe(cm_df)

        # Classification report
        st.markdown("**Classification report (test set):**")
        report = classification_report(y_test, y_test_pred, output_dict=False)
        st.text(report)

    # ----------------------------------------------------------------
    # MODELING: DEATH
    # ----------------------------------------------------------------
    elif view == "Modeling: DEATH":
        st.subheader("Modeling – All-cause Mortality (Period 1)")

        st.markdown("### 1. Analytic dataset overview")
        st.write("Shape of incident DEATH dataset (rows, columns):", incident_death_df.shape)
        st.write("Columns:")
        st.write(list(incident_death_df.columns))

        st.write("Class balance (DEATH = 0/1):")
        st.dataframe(incident_death_df["DEATH"].value_counts().rename("Count").to_frame())

        st.markdown("### 2. Train/test split and standardization")

        # Prepare X, y
        X_death, y_death = prepare_model_data(incident_death_df, outcome_col="DEATH")

        st.write("Predictor matrix shape:", X_death.shape)
        st.write("Outcome vector length:", len(y_death))

        # Train/test split + scaling
        (
            X_train_d,
            X_test_d,
            y_train_d,
            y_test_d,
            X_train_scaled_d,
            X_test_scaled_d,
            scaler_death,
        ) = train_test_and_scale(X_death, y_death, test_size=0.3, random_state=42)

        st.write("Training set size:", X_train_d.shape[0])
        st.write("Test set size:", X_test_d.shape[0])

        with st.expander("Show first 5 rows of original vs standardized predictors (train set)"):
            st.write("Original X_train (head):")
            st.dataframe(X_train_d.head())
            st.write("Standardized X_train (head):")
            st.dataframe(X_train_scaled_d.head())

        st.markdown("### 3. Correlation heatmap of predictors")
        fig_corr_d = plot_corr_heatmap(X_death, "Correlation Heatmap – DEATH Predictors")
        st.pyplot(fig_corr_d)

        st.markdown("### 4. Logistic regression model – DEATH")

        model_death = LogisticRegression(solver="liblinear")
        model_death.fit(X_train_scaled_d, y_train_d)

        y_train_pred_d = model_death.predict(X_train_scaled_d)
        y_test_pred_d = model_death.predict(X_test_scaled_d)

        train_acc_d = accuracy_score(y_train_d, y_train_pred_d)
        test_acc_d = accuracy_score(y_test_d, y_test_pred_d)

        st.write(f"Training accuracy: **{train_acc_d:.3f}**")
        st.write(f"Test accuracy: **{test_acc_d:.3f}**")

        cm_d = confusion_matrix(y_test_d, y_test_pred_d)
        cm_d_df = pd.DataFrame(
            cm_d,
            index=["Actual 0", "Actual 1"],
            columns=["Pred 0", "Pred 1"],
        )
        st.markdown("**Confusion matrix (test set):**")
        st.dataframe(cm_d_df)

        st.markdown("**Classification report (test set):**")
        report_d = classification_report(y_test_d, y_test_pred_d, output_dict=False)
        st.text(report_d)



if __name__ == "__main__":
    main()







