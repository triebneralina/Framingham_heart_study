import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
    inconsistent_data = df[(df["CURSMOKE"] == 0) & (df["CIGPDAY"] > 0)]
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


def plot_distribution(df, col):
    figs = []

    # Histogram
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    sns.histplot(df[col].dropna(), kde=True, bins=30, ax=ax1)
    ax1.set_title(f"Distribution of {col}")
    ax1.set_xlabel(col)
    ax1.set_ylabel("Frequency")
    fig1.tight_layout()
    figs.append(fig1)

    # Boxplot
    fig2, ax2 = plt.subplots(figsize=(6, 2))
    sns.boxplot(x=df[col].dropna(), ax=ax2)
    ax2.set_title(f"Box Plot of {col}")
    ax2.set_xlabel(col)
    fig2.tight_layout()
    figs.append(fig2)

    return figs


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
            "Consistency check: Smoking",
            "Missing data",
            "Distributions (risk profile)",
            "Winsorization summary",
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

    # ----------------------------------------------------------------
    # OUTCOMES BY SEX
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
    # SMOKING CONSISTENCY
    # ----------------------------------------------------------------
    elif view == "Consistency check: Smoking":
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
    # DISTRIBUTIONS 
    # ----------------------------------------------------------------
    elif view == "Distributions (risk profile)":
        st.subheader(f"Distributions of Risk Profile Variables - Period 1")

        risk_df = make_risk_profile_df(period1_df)
        numeric_cols = [
            c for c in RISK_PROFILE_COLUMNS
            if c in risk_df.columns and pd.api.types.is_numeric_dtype(risk_df[c])
        ]

        if not numeric_cols:
            st.warning("No numeric risk profile columns found for this period.")
        else:
            col_choice = st.selectbox("Select numeric column to plot", numeric_cols)

            fig1, ax1 = plt.subplots(figsize = (8,4))

            sns.histplot(
                data=risk_df,
                x=col_choice,
                hue="SEX",
                multiple="dodge",  # side-by-side bars
                palette=["paleturquoise", "pink"],
                alpha=1.0,
                kde=True,
                ax=ax1
            )

            ax1.set_title(f"{col_choice} Distribution by SEX")
            ax1.set_xlabel(col_choice)
            ax1.set_ylabel("Count")
            
            fig1.tight_layout()
            st.pyplot(fig1)

            #------------------BOX PLOT------------------------------
            fig2, ax2 = plt.subplots(figsize=(6, 3))
            sns.boxplot(
                data=risk_df,
                x="SEX",
                y=col_choice,
                palette=["paleturquoise", "pink"],
                ax=ax2
            )
            ax2.set_title(f"Box Plot of {col_choice} by Sex", fontsize=12, fontweight="bold")
            ax2.set_xlabel("SEX")
            ax2.set_ylabel(col_choice)

            fig2.tight_layout()
            st.pyplot(fig2)

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


if __name__ == "__main__":
    main()







