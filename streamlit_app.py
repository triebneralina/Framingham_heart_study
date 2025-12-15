import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import roc_curve, auc, RocCurveDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

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

def prev_consistency_checks(df):
    """
    Returns a dictionary of inconsistency dataframes for PREV* logical checks.
    """
    prev_cols = ["PREVCHD", "PREVAP", "PREVMI", "PREVSTRK"]
    present = [c for c in prev_cols if c in df.columns]

    results = {}

    # 1) Non-binary values check
    non_binary = pd.DataFrame()
    for c in present:
        bad = df[~df[c].dropna().isin([0, 1])]
        if not bad.empty:
            non_binary = bad[[c]].copy()
            break
    results["Non-binary PREV* values"] = non_binary

    # 2) PREVCHD == 0 but PREVMI == 1 or PREVAP == 1
    if "PREVCHD" in df.columns and ("PREVMI" in df.columns or "PREVAP" in df.columns):
        cond = (df["PREVCHD"] == 0) & (
            (df["PREVMI"] == 1 if "PREVMI" in df.columns else False)
            | (df["PREVAP"] == 1 if "PREVAP" in df.columns else False)
        )
        results["PREVCHD=0 but PREVMI or PREVAP = 1"] = df.loc[cond, ["PREVCHD", "PREVMI", "PREVAP"]].copy()

    # 3) PREVCHD == 1 but PREVMI == 0 and PREVAP == 0 (suspicious)
    if "PREVCHD" in df.columns and "PREVMI" in df.columns and "PREVAP" in df.columns:
        cond = (df["PREVCHD"] == 1) & (df["PREVMI"] == 0) & (df["PREVAP"] == 0)
        results["PREVCHD=1 but PREVMI=0 and PREVAP=0"] = df.loc[cond, ["PREVCHD", "PREVMI", "PREVAP"]].copy()


    return results


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
        color = "paleturquoise"
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

def plot_outcome_cases_by_sex(df, outcome_col, title):
    """
    Bar chart of outcome==1 cases by sex, plus a summary table
    (counts and percentages within sex).
    Assumes SEX coded as 1=Male, 2=Female.
    """
    plt.style.use("ggplot")

    # Filter to cases where outcome == 1
    cases = df[df[outcome_col] == 1].copy()

    # Count cases by sex (1=Male, 2=Female), ensure both present
    by_sex = cases["SEX"].value_counts().reindex([1, 2]).fillna(0).astype(int)

    # Bar chart
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = by_sex.index.map({1: "Male", 2: "Female"})
    bars = ax.bar(labels, by_sex.values, color=["paleturquoise", "pink"])

    # Add counts on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"n = {int(height)}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Sex")
    ax.set_ylabel(f"Number of {outcome_col} = 1")
    fig.tight_layout()

    # Build summary table: totals, cases, no-cases, percentages, diff check
    rows = []
    sex_label_map = {1: "Male", 2: "Female"}
    for sex_code in [1, 2]:
        label = sex_label_map.get(sex_code, str(sex_code))

        total = (df["SEX"] == sex_code).sum()
        cases_sex = ((df["SEX"] == sex_code) & (df[outcome_col] == 1)).sum()
        no_cases_sex = ((df["SEX"] == sex_code) & (df[outcome_col] == 0)).sum()
        diff = total - (cases_sex + no_cases_sex)
        pct = (cases_sex / total * 100) if total > 0 else np.nan

        rows.append(
            {
                "Sex": label,
                "Total": total,
                f"{outcome_col} = 1 (cases)": cases_sex,
                f"{outcome_col} = 0 (non-cases)": no_cases_sex,
                "Cases %": pct,
                "Check (Total - (cases + non-cases))": diff,
            }
        )

    summary_df = pd.DataFrame(rows)

    return fig, summary_df


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

#--------------------------------------------------------------------
# MODELS
#--------------------------------------------------------------------
def get_models(random_state=42):
    return {
        "LogReg (balanced)": LogisticRegression(
            solver="liblinear",  #We used the liblinear solver because it is well suited for binary logistic regression with moderate sample sizes and class imbalance.
            class_weight="balanced",
            max_iter=1000,
        ),
        "Random Forest (balanced)": RandomForestClassifier(
            n_estimators=300,
            random_state=random_state,
            class_weight="balanced",
        ),
        "SVM (balanced)": SVC(
            kernel="rbf",
            class_weight="balanced",
            probability=True,   # needed for ROC/AUC
            random_state=random_state,
        ),
        "Neural Net (MLP)": MLPClassifier(
            hidden_layer_sizes=(32, 16),   
            activation="relu",
            solver="adam",
            alpha=0.001,                  
            max_iter=200,
            early_stopping=True,           # stops if validation score stops improving
            random_state=random_state,
        ),            
    }


def run_cross_validation(models, X, y, n_splits=5, random_state=42):
    """
    Returns a dataframe of mean CV scores for each model.
    Uses stratified k-fold and multiple metrics like in Colab.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    rows = []
    for name, model in models.items():
        scores = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        rows.append(
            {
                "Model": name,
                "CV Accuracy (mean)": np.mean(scores["test_accuracy"]),
                "CV Precision (mean)": np.mean(scores["test_precision"]),
                "CV Recall (mean)": np.mean(scores["test_recall"]),
                "CV F1 (mean)": np.mean(scores["test_f1"]),
                "CV ROC AUC (mean)": np.mean(scores["test_roc_auc"]),
            }
        )

    return pd.DataFrame(rows).sort_values(by="CV ROC AUC (mean)", ascending=False)


def fit_models_and_eval(models, X_train_scaled, X_test_scaled, y_train, y_test):
    """
    Fits each model on train, evaluates on test.
    Returns:
      results_df, confusion_matrices_dict, classification_reports_dict, roc_data_dict
    """
    results = []
    conf_mats = {}
    reports = {}
    roc_data = {}  # name -> (fpr, tpr, auc)

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        test_acc = accuracy_score(y_test, y_pred)

        conf_mats[name] = confusion_matrix(y_test, y_pred)
        reports[name] = classification_report(y_test, y_pred, output_dict=False)

        # ROC (need probabilities)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            roc_data[name] = (fpr, tpr, roc_auc)

        results.append({"Model": name, "Test Accuracy": test_acc})

    results_df = pd.DataFrame(results).sort_values(by="Test Accuracy", ascending=False)
    return results_df, conf_mats, reports, roc_data

def plot_confusion_matrix_heatmap(cm, title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        linewidths=0.5,
        linecolor="white",
        xticklabels=["Pred 0", "Pred 1"],
        yticklabels=["Actual 0", "Actual 1"],
        ax=ax,
    )
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    fig.tight_layout()
    return fig

def plot_roc_comparison(roc_data, title):
    """Overlay ROC curves for multiple models."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, (fpr, tpr, roc_auc) in roc_data.items():
        ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    fig.tight_layout()
    return fig

def predict_with_threshold(model, X, threshold=0.5):
    """
    Returns y_prob (P(class=1)) and y_pred based on a custom threshold.
    Requires model to support predict_proba.
    """
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    return y_prob, y_pred



# -------------------------------------------------------------------
# MAIN APP
# -------------------------------------------------------------------

def main():
    st.title("Framingham Heart Study: CVD Risk Analysis by Sex")
    st.write("Interactive report on the loading, cleaning and analysis of the Framingham dataset")

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
            "Research question",
            "Raw data overview",
            "Risk profile by sex",
            "Outcomes by sex",
            "Consistency checks",
            "Missing data",
            "Winsorization summary",
            "Modeling: CVD",
            "Modeling: DEATH",
            "Conclusion",
        ],
    )

    #----------------------------------------------------------------
    # RESEARCH QUESTION
    #----------------------------------------------------------------
    if view == "Research question":
        st.subheader("Research Question")
        st.markdown("PICOT formatted research question:")
        st.info("In adults in the Framingham Heart Study (P), "\
            "how well do baseline cardiovascular risk factors, including sex (I), " \
            "predict incident cardiovascular events and all-cause mortality over 24 years (O, T), "
            "and does predictive performance or estimated risk differ between men and women (C)? ")
        
        st.subheader("Aim")
        st.markdown("""
                    **The aims of this analysis are to:**
                    1. Describe gender differences in risk factors
                    2. Build predictive models for selected outcomes
                    """)

    
    # ----------------------------------------------------------------
    # RAW DATA OVERVIEW
    # ----------------------------------------------------------------
    elif view == "Raw data overview":
        st.subheader("Raw Data: Period 1")
        st.info("""
                We restrict all analyses to **Period 1**.

        We were given a dataset spans examinations conducted between 1956 to 1968. These examinations happened in exam cycles of the **Original Framingham Heart Study cohort**.
        Therefore, this data does not capture the true cohort baseline established at the first Framingham examination in 1948. 
        As a result, some participants may already have experienced cardiovascular events prior to "our" Period 1 examination.
        To ensure a valid and interpretable baseline for our research question, we treat Period 1 as the analytic baseline and apply additional exclusion criteria when examining incident cardiovascular outcomes 
        (e.g., excluding participants with prevalent disease at Period 1).

        Including data from subsequent exam periods (Periods 2 and 3) would introduce several issues, including:
                
        - Overlapping participants with different baseline times,
        - Inconsistent availability of key risk factor measurements across exam cycles, 
        - Inconsistencies in defining the population “at risk” for incident outcomes.

        Restricting the analysis to Period 1 ensures that all participants:
        - Are observed at a common baseline time point,
        - Have uniformly recorded baseline risk factors,
        - Are followed prospectively for outcomes using the same time origin.

        **Benefits:**
        - Clearly defined and consistent baseline for risk factor assessment
        - Uniform measurement availability across participants
        - Valid estimation of incident cardiovascular outcomes

        **Limitations:**
        - Reduces sample size, potentially lowering statistical power and increasing variance
        - Limits generalizability to modern populations, as the cohort reflects mid-20th-century risk profiles and clinical practices        
       """)

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
    # RISK PROFILE BY SEX: 
    # ----------------------------------------------------------------
    elif view == "Risk profile by sex":
        st.subheader(f"Risk Profile by Sex: Period 1")

        risk_df = make_risk_profile_df(period1_df)

        st.write("**Selected risk profile columns:**")
        st.write(RISK_PROFILE_COLUMNS)
        st.info("These variables were selected based on their established relevance to cardiovascular disease risk and availability in Period 1."
        " Only variables measures simultaneously at baseline were included, as this ensures an ordered relationship between predictors and outcomes."
        )

        if risk_df.empty:
            st.warning("No risk profile columns found for this period.")
        else:
            # Choose a variable- avoids table with 120 columns
            variable = st.selectbox(
                "Select a variable to view summary statistics:",
                RISK_PROFILE_COLUMNS
            )

            st.subheader(f"Summary Statistics for {variable} by Sex")

            if variable not in risk_df.columns:
                st.warning(f"{variable} is not in the dataframe.")
            else:
                # Group by sex for selected variable
                grouped = risk_df.groupby("SEX")[variable]

                # Compute descriptive statistics
                stats = grouped.describe().T

                # Map SEX codes to readable column names
                stats.columns = [
                    "Male" if col == 1 else "Female" if col == 2 else str(col)
                    for col in stats.columns
                ]

                # Optional: nicer index name
                stats.index.name = "Statistic"

                st.dataframe(stats)

            # (Optional) keep this if you still want to see some raw rows
            with st.expander("Show first rows grouped by sex"):
                st.dataframe(
                    risk_df.groupby("SEX").head().sort_values(by="SEX")
                )


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

        st.subheader("Discussion")
        st.info("""
                Here we looked at the risk profile distributions for males and females as a part of EDA. At this stage, we noticed some things:
                - Box plots show  good amount of outliers, we will address this later 
                - When checking the CIGPDAY feature, there was a peak for n=20. This is the number of cigarettes in a pack, and could reflect some rounding down / up which could introduce bias 
                - More females were being treated with BPMEDS than men. This reflects baseline differences in treatment (rather than risk), a potential confounding factor.
                - In all previous incidences, males had more incidences than women for all
                """)

    # ----------------------------------------------------------------
    # OUTCOMES BY SEX: 
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

    # ----------Bar Plots------------------------------------------------------
        st.subheader("Visualization of outcomes")

        outcomes_df = make_outcomes_df(period1_df)

        st.write("**Outcome columns used:**")
        st.write(OUTCOME_COLUMNS)

        if outcomes_df.empty:
            st.warning("No outcome columns found for this period.")
        else:
            # --- CVD ---
            if "CVD" in outcomes_df.columns:
                st.markdown("### CVD cases by sex")

                fig_cvd, summary_cvd = plot_outcome_cases_by_sex(
                    outcomes_df,
                    outcome_col="CVD",
                    title="CVD Cases by Sex (Period 1)",
                )
                st.pyplot(fig_cvd)

                st.write("**CVD counts & percentages by sex:**")
                st.dataframe(summary_cvd)
            else:
                st.info("CVD column not found in the dataset.")

            st.markdown("---")

            # --- DEATH ---
            if "DEATH" in outcomes_df.columns:
                st.markdown("### Deaths by sex")

                fig_death, summary_death = plot_outcome_cases_by_sex(
                    outcomes_df,
                    outcome_col="DEATH",
                    title="Deaths by Sex (Period 1)",
                )
                st.pyplot(fig_death)

                st.write("**Death counts & percentages by sex:**")
                st.dataframe(summary_death)
            else:
                st.info("DEATH column not found in the dataset.")
        

        #-----------------Outcome event rates--------------------------------
        st.subheader("Overall Outcome Event Counts and Percentages")

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

        st.subheader("Discussion")
        st.info("""
                These distributions differ clearly from what we know today:
                - The sex gap is much larger (around a 16% difference) than the one observed today.
                - Nowadays most modern studies show an increased incidence of CVD and CVD-related deaths than men - the opposite is shown by the data

                We believe this difference reflects historical context:
                - CVD diagnosis was based on male expression of the diseases, leading to women being under-diagnosed in that time. This explains women seeming more protected in our data.
                - CVD-related deaths are less prevalent nowadays due to modern medicine being much more effective, as well as the recognition of sex differences in disease
                """)


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
        st.markdown("---")
        st.subheader("PREV* consistency checks (baseline medical history)")
        st.info("""
                We performed a coherence check on the binary variables PrevCHD, PrevMI, and PrevAP.
                PrevCHD was required to be consistent with PrevMI and PrevAP (since it is a combined variable for those 2).
                If either PrevMI or PrevAP equaled 1, then PrevCHD had to equal 1.
                
                Our initial reasoning was that, conversely
                PrevCHD could only equal 0 when both PrevMI and PrevAP equaled 0. When we found some inconsistencies, we re-examined this premise.
                Since coronary heart disease is am umbrella term for multiple pathologies, it is possible that these inconsistencies stem from a "different" coronary heart disease.
                Alternatively, inconsistencies in recording practices for this variable could account for the inconsistency. This is why we decided to keep the inconsistent rows.
                """)
                 
        prev_checks = prev_consistency_checks(period1_df)

        for check_name, bad_rows in prev_checks.items():
            st.markdown(f"### {check_name}")
            if bad_rows.empty:
                st.success("No inconsistencies found.")
            else:
                st.error(f"Found {len(bad_rows)} inconsistent/suspicious rows.")
                st.dataframe(bad_rows.head(20))


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

        st.subheader("Dealing with the missing values")
        st.info("""
                The features that had missing values never had a missingness percentage higher than 5% so we dropped these. 
                Our dataset is small to begin with, so data imputation would not be smart because it risks introducing data leakage.
                """)

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
        phys_cols_present = [c for c in PHYSIOLOGICAL_LIMITS.keys() if c in dropped_df.columns]
        
        before_stats = dropped_df[list(PHYSIOLOGICAL_LIMITS.keys())].describe()

        winsorized_df = winsorize_period1(dropped_df)
        after_stats = winsorized_df[list(PHYSIOLOGICAL_LIMITS.keys())].describe()

        st.markdown("### Descriptive Statistics Before Winsorization")
        st.dataframe(before_stats)

        st.info("""
                We addressed outliers by selecting physiological limits and winsorising at these values for numerical risk factors. 
                These were the ranges we accepted, anything above or below was winsorised:
                
                - Age: 18 - 110 years
                
                - Cigpday: < 80 (80 would represent chain smoking)
                
                - BMI: 10 - 70 kg/m^2
                
                - SYSBP: 60 - 300 mmHg
                
                - DIASBP: 30 - 150 mmHg
                
                - HR: 30 - (220 - 32)bpm (220bpm minus the youngest age)
                
                - TOTCHOL: 70 - 600 mg/dL (600mg/dL represents extreme familial hypercholesteremia)
                
                Then we visualised our descriptive statistics and distributions following outlier handling.
                """)



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

    #-----------------------------------------------------------------
    # VISUALIZATION AFTER WINSORIZING
    #-----------------------------------------------------------------

    #-----------------Check for remaining missings----------------------
        st.markdown("### Check for Remaining Missing Values (After Dropping + Winsorization)")

        missing_after = compute_missing_info(winsorized_df[phys_cols_present])

        if missing_after.empty:
            st.success("No missing values remain in the physiological variables after dropping and winsorization.")
        else:
            st.error("Some missing values are still present after cleaning:")
            st.dataframe(missing_after)

    #-----------------Distribution check before vs after------------------------
        st.markdown("### Distributions Before vs After Winsorization")

        if not phys_cols_present:
            st.warning("No physiological variables found for winsorization.")
        else:
            col_choice = st.selectbox(
                "Select physiological variable to compare:",
                phys_cols_present,
            )

            # Histograms before vs after
            fig_hist, axes = plt.subplots(1, 2, figsize=(10, 4))

            sns.histplot(
                dropped_df[col_choice].dropna(),
                kde=True,
                bins=30,
                ax=axes[0],
                color="grey",
            )
            axes[0].set_title(f"{col_choice} – Before winsorization")
            axes[0].set_xlabel(col_choice)
            axes[0].set_ylabel("Count")

            sns.histplot(
                winsorized_df[col_choice].dropna(),
                kde=True,
                bins=30,
                ax=axes[1],
                color="green",
            )
            axes[1].set_title(f"{col_choice} – After winsorization")
            axes[1].set_xlabel(col_choice)
            axes[1].set_ylabel("Count")

            fig_hist.tight_layout()
            st.pyplot(fig_hist)

            # Boxplots before vs after
            fig_box, axes_box = plt.subplots(1, 2, figsize=(10, 3))

            sns.boxplot(
                x=dropped_df[col_choice].dropna(),
                ax=axes_box[0],
                color="grey",
            )
            axes_box[0].set_title(f"{col_choice} – Before winsorization")
            axes_box[0].set_xlabel(col_choice)

            sns.boxplot(
                x=winsorized_df[col_choice].dropna(),
                ax=axes_box[1],
                color="green",
            )
            axes_box[1].set_title(f"{col_choice} – After winsorization")
            axes_box[1].set_xlabel(col_choice)

            fig_box.tight_layout()
            st.pyplot(fig_box)    

    # ----------------------------------------------------------------
    # MODELING: CVD
    # ----------------------------------------------------------------
    elif view == "Modeling: CVD":
        st.subheader("Modeling: Incident CVD (Period 1)")

        st.markdown("### 1. Analytic dataset overview")
        st.write("Shape of incident CVD dataset (rows, columns):", incident_cvd_df.shape)

        st.info("""
        Our introduction (see raw data overview), touches on the reasoning behind the exclusion of previous disease incidence (PREVCHD, PREVSTRK) when examining CVD. 
        This exclusion ensures that all individuals included in the analysis were genuinely at risk of experiencing a first CVD event at the start of follow-up.
                
        Including participants with pre-existing CVD would conflate disease prevalence with incidence, as such individuals are no longer susceptible to a first occurrence of the 
        outcome and instead represent a population at risk for recurrence.
        This would bias estimates of incident risk and obscure associations between baseline risk factors and new-onset disease.
                
        Restricting the analytic cohort to participants free of CVD at baseline aligns with standard epidemiologic practice and allows for valid estimation of incident CVD risk using baseline predictors.
                """)
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

        st.markdown("### 4. Baseline Logistic regression model CVD (Unbalanced)")
        st.caption( "This baseline model uses standard logistic regression without class balancing," \
        "serving as a reference point for comparison with more advanced models.")

        # Fit logistic regression on standardized predictors
        model_cvd = LogisticRegression(solver="liblinear", max_iter = 1000)
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
  
        st.markdown("**Confusion matrix (test set):**")
        fig_cm = plot_confusion_matrix_heatmap(cm, title="Confusion Matrix")
        st.pyplot(fig_cm)


        # Classification report
        st.markdown("**Classification report (test set):**")
        report_dict = classification_report(y_test, y_test_pred, output_dict=True)
        report_df = pd.DataFrame(report_dict).T
        st.dataframe(report_df.style
                        .format("{:.2f}")
                        .background_gradient(cmap = "Blues", subset = ["precision", "recall", "f1-score"])
                        )

        st.markdown("### 5. Compare models (LogReg, Random Forest, SVM, NN)")

        models = get_models(random_state=42)

        # --- Cross-validation on TRAINING DATA ONLY (best practice) ---
        st.markdown("#### 5-fold Cross-Validation (on training set)")
        cv_summary = run_cross_validation(models, X_train_scaled, y_train, n_splits=5, random_state=42)
        st.dataframe(cv_summary)

        # --- Fit models + evaluate on test set ---
        st.markdown("#### Test Set Performance")
        results_df, conf_mats, reports, roc_data = fit_models_and_eval(
            models,
            X_train_scaled, X_test_scaled,
            y_train, y_test
        )
        st.dataframe(results_df)

        # Choose a model to inspect details
        model_choice = st.selectbox("Select model for detailed metrics:", list(models.keys()), key="cvd_model_select")

        st.markdown("**Confusion matrix (test set):**")
        cm = conf_mats[model_choice]
        fig_cm = plot_confusion_matrix_heatmap(cm, title="Confusion Matrix")
        st.pyplot(fig_cm)


        st.markdown("**Classification report (test set):**")
        report_dict = classification_report(y_test, y_test_pred, output_dict=True)
        report_df = pd.DataFrame(report_dict).T
        st.dataframe(report_df.style
                        .format("{:.2f}")
                        .background_gradient(cmap = "Blues", subset = ["precision", "recall", "f1-score"])
                        )

        # ROC comparison plot (only models with predict_proba)
        if roc_data:
            st.markdown("#### ROC Curves (test set)")
            fig_roc = plot_roc_comparison(roc_data, "ROC Curve Comparison – CVD")
            st.pyplot(fig_roc)
        else:
            st.info("No ROC data available (models missing predict_proba).")

            st.markdown("### Threshold tuning (test set)")

        chosen_model = models[model_choice]

        if not hasattr(chosen_model, "predict_proba"):
            st.info("This model does not support probability outputs (predict_proba). Threshold tuning unavailable.")
        else:
            threshold = st.slider(
                "Select probability threshold for class=1",
                min_value=0.05,
                max_value=0.95,
                value=0.50,
                step=0.05,
                key="cvd_threshold_slider",
            )

            y_prob, y_pred_thr = predict_with_threshold(chosen_model, X_test_scaled, threshold=threshold)

            acc_thr = accuracy_score(y_test, y_pred_thr)
            cm_thr = confusion_matrix(y_test, y_pred_thr)

            st.write(f"Accuracy at threshold {threshold:.2f}: **{acc_thr:.3f}**")

            st.markdown("**Confusion matrix (threshold tuned):**")
            st.dataframe(
                pd.DataFrame(
                    cm_thr,
                    index=["Actual 0", "Actual 1"],
                    columns=["Pred 0", "Pred 1"],
                )
            )
            fig_cm_thr = plot_confusion_matrix_heatmap(cm_thr, title="Confusion Matrix (Threshold tuned)")
            st.pyplot(fig_cm_thr)


            st.markdown("**Classification report (threshold tuned):**")
            report_dict = classification_report(y_test, y_pred_thr, output_dict=True)
            report_df = pd.DataFrame(report_dict).T
            st.dataframe(report_df.style
                        .format("{:.2f}")
                        .background_gradient(cmap = "Blues", subset = ["precision", "recall", "f1-score"])
                        )   

            st.subheader("Discussion")
            st.info("""
                    **Classical models (logistic regression, random forest, SVM):**

                    Hyperparameter tuning and decision threshold optimisation were explored but resulted in negligible changes to performance metrics. Accuracy and related measures remained largely unchanged, so further optimisation results are not reported.
                    
                    **Neural networks:**
                    
                    Neural network models performed poorly and showed high variability across runs, likely due to the small dataset size and the data-intensive nature of these models.
                    
                    **Model stability:**
                    Neural network outcomes fluctuated substantially between extreme and more plausible predictions, indicating sensitivity to random initialisation and limited robustness.
                    """)      

    # ----------------------------------------------------------------
    # MODELING: DEATH
    # ----------------------------------------------------------------
    elif view == "Modeling: DEATH":
        st.subheader("Modeling: All-cause Mortality (Period 1)")

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
        fig_corr_d = plot_corr_heatmap(X_death, "Correlation Heatmap: DEATH Predictors")
        st.pyplot(fig_corr_d)

        st.markdown("### 4. Baseline Logistic regression model DEATH (Unbalanced)")

        model_death = LogisticRegression(solver="liblinear")
        model_death.fit(X_train_scaled_d, y_train_d)

        y_train_pred_d = model_death.predict(X_train_scaled_d)
        y_test_pred_d = model_death.predict(X_test_scaled_d)

        train_acc_d = accuracy_score(y_train_d, y_train_pred_d)
        test_acc_d = accuracy_score(y_test_d, y_test_pred_d)

        st.write(f"Training accuracy: **{train_acc_d:.3f}**")
        st.write(f"Test accuracy: **{test_acc_d:.3f}**")

        st.markdown("**Confusion matrix (test set):**")
        
        cm_d = confusion_matrix(y_test_d, y_test_pred_d)
        fig_cm_d_df = plot_confusion_matrix_heatmap(cm_d, title="Confusion Matrix")
        st.pyplot(fig_cm_d_df)

        st.markdown("**Classification report (test set):**")            
        report_d = classification_report(y_test_d, y_test_pred_d, output_dict=True)
        report_df = pd.DataFrame(report_d).T
        st.dataframe(report_df.style
                        .format("{:.2f}")
                        .background_gradient(cmap = "Blues", subset = ["precision", "recall", "f1-score"])
                        )     
        
        st.markdown("### 5. Compare models (LogReg, Random Forest, SVM, NN)")

        models = get_models(random_state=42)

        st.markdown("#### 5-fold Cross-Validation (on training set)")
        cv_summary = run_cross_validation(models, X_train_scaled_d, y_train_d, n_splits=5, random_state=42)
        st.dataframe(cv_summary)

        st.markdown("#### Test Set Performance")
        results_df, conf_mats, reports, roc_data = fit_models_and_eval(
            models,
            X_train_scaled_d, X_test_scaled_d,
            y_train_d, y_test_d
        )
        st.dataframe(results_df)

        model_choice = st.selectbox("Select model for detailed metrics:", list(models.keys()), key="death_model_select")

        st.markdown("**Confusion matrix (test set):**")
        cm = conf_mats[model_choice]

        fig_cm = plot_confusion_matrix_heatmap(cm, title="Confusion Matrix")
        st.pyplot(fig_cm)

        st.markdown("**Classification report (test set):**")
        st.dataframe(report_df.style
                        .format("{:.2f}")
                        .background_gradient(cmap = "Blues", subset = ["precision", "recall", "f1-score"])
                        )

        if roc_data:
            st.markdown("#### ROC Curves (test set)")
            fig_roc = plot_roc_comparison(roc_data, "ROC Curve Comparison – DEATH")
            st.pyplot(fig_roc)
        else:
            st.info("No ROC data available (models missing predict_proba).")

        st.markdown("### Threshold tuning (test set)")

        chosen_model = models[model_choice]

        if not hasattr(chosen_model, "predict_proba"):
            st.info("This model does not support probability outputs (predict_proba). Threshold tuning unavailable.")
        else:
            threshold = st.slider(
                "Select probability threshold for class=1",
                min_value=0.05,
                max_value=0.95,
                value=0.50,
                step=0.05,
                key="death_threshold_slider",
            )

            y_prob, y_pred_thr = predict_with_threshold(chosen_model, X_test_scaled_d, threshold=threshold)

            acc_thr = accuracy_score(y_test_d, y_pred_thr)
            cm_thr = confusion_matrix(y_test_d, y_pred_thr)

            st.write(f"Accuracy at threshold {threshold:.2f}: **{acc_thr:.3f}**")

            st.markdown("**Confusion matrix (threshold tuned):**")
            fig_cm_thr = plot_confusion_matrix_heatmap(cm_thr, title="Confusion Matrix")
            st.pyplot(fig_cm_thr)

            st.markdown("**Classification report (threshold tuned):**")
            report_d = classification_report(y_test_d, y_pred_thr, output_dict=True)    
            report_df = pd.DataFrame(report_d).T
            st.dataframe(report_df.style
                        .format("{:.2f}")
                        .background_gradient(cmap = "Blues", subset = ["precision", "recall", "f1-score"])
                        )  

            st.subheader("Discussion")
            st.info("""
                    **Classical models (logistic regression, random forest, SVM):**

                    Hyperparameter tuning and decision threshold optimisation were explored but resulted in negligible changes to performance metrics. Accuracy and related measures remained largely unchanged, so further optimisation results are not reported.
                    
                    **Neural networks:**
                    
                    Neural network models performed poorly and showed high variability across runs, likely due to the small dataset size and the data-intensive nature of these models.
                    
                    **Model stability:**
                    Neural network outcomes fluctuated substantially between extreme and more plausible predictions, indicating sensitivity to random initialisation and limited robustness.
                    """)       
    #--------------------------------------------------------------------------------------------------------------
    # CONCLUSION
    #---------------------------------------------------------------------------------------------------------------
    elif view == "Conclusion":
        st.subheader("Conclusion")
        st.write("""
                 Our RQ: “In adults in the Framingham Heart Study (P), how well do baseline cardiovascular risk factors, 
                 including sex (I), predict incident cardiovascular events and all-cause mortality over 24 years (O, T), 
                 and does predictive performance or estimated risk differ between men and women (C)?”
                
                 To answer this question, we began by performing exploratory data analysis, visualising and using descriptive 
                 statistics to observe the risk profiles and outcomes by sex in Period 1.
                 We then selected features relevant to our RQ (risk factors and outcomes CVD as well as death).
                 We then created incident datarames for our outcomes (CVD and death). For the CVD dataframe, we filtered out 
                 previous incidents of CVD using PREV* variables. The dataframe for Death did not need this exclusion. 
                 Total 2 data frames, 1 for each outcome: this prevents data leakage as previous incidences are very highly 
                 correlated with their respective outcomes. 
                
                 Next, we cleaned the data, this involved:
                - Identifying missing values and dropping them (small proportion)
                - Handling outliers by applying physiological thresholds and winsorising extreme outliers.
                - Following this, we checked for erroneous data/inconsistencies between variables like CURSMOKE and CIGPDAY, as well as all PREV- variables. 
                We plotted our data again to observe changes before and after our cleaning.
                 
                 Before Standardisation, we performed the train/test split for both our CVD and Death df.
                 Then, we standardised our data by (I'm not sure what we did here. Can someone write this, @Seb?)
                 
                Our data was then ready to train 3 different ML models:
                1. Logistic regression (CVD and Death separately)
                2. Random forest (CVD and Death separately)
                3. Support Vector Machine (CVD and Death separately)
                (We attempted to train neural networks but our dataset was too small to give good results)

                 Finally, we applied k-fold analysis to all our models (6 in total) to assess performance more robustly.

                Our best model was _______, which could be explained by______, however the models did not perform well enough to say that one of them was particularly well performing. We indeed saw a difference in risk factors and outcomes by sex, however historical contexts outline how these differences are not consistent with modern knowledge.
                 """)

if __name__ == "__main__":
    main()









# --- merged from import_streamlit_as_st.py.bak ---
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import roc_curve, auc, RocCurveDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

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

def prev_consistency_checks(df):
    """
    Returns a dictionary of inconsistency dataframes for PREV* logical checks.
    """
    prev_cols = ["PREVCHD", "PREVAP", "PREVMI", "PREVSTRK"]
    present = [c for c in prev_cols if c in df.columns]

    results = {}

    # 1) Non-binary values check
    non_binary = pd.DataFrame()
    for c in present:
        bad = df[~df[c].dropna().isin([0, 1])]
        if not bad.empty:
            non_binary = bad[[c]].copy()
            break
    results["Non-binary PREV* values"] = non_binary

    # 2) PREVCHD == 0 but PREVMI == 1 or PREVAP == 1
    if "PREVCHD" in df.columns and ("PREVMI" in df.columns or "PREVAP" in df.columns):
        cond = (df["PREVCHD"] == 0) & (
            (df["PREVMI"] == 1 if "PREVMI" in df.columns else False)
            | (df["PREVAP"] == 1 if "PREVAP" in df.columns else False)
        )
        results["PREVCHD=0 but PREVMI or PREVAP = 1"] = df.loc[cond, ["PREVCHD", "PREVMI", "PREVAP"]].copy()

    # 3) PREVCHD == 1 but PREVMI == 0 and PREVAP == 0 (suspicious)
    if "PREVCHD" in df.columns and "PREVMI" in df.columns and "PREVAP" in df.columns:
        cond = (df["PREVCHD"] == 1) & (df["PREVMI"] == 0) & (df["PREVAP"] == 0)
        results["PREVCHD=1 but PREVMI=0 and PREVAP=0"] = df.loc[cond, ["PREVCHD", "PREVMI", "PREVAP"]].copy()


    return results


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
# PLOTTING HELPERS:
# -------------------------------------------------------------------
def plot_missing_bar(missing_info, title):
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(
        x=missing_info.index,
        y="Missing Percentage",
        data=missing_info,
        ax=ax,
        color = "paleturquoise"
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

def plot_outcome_cases_by_sex(df, outcome_col, title):
    """
    Bar chart of outcome==1 cases by sex, plus a summary table
    (counts and percentages within sex).
    Assumes SEX coded as 1=Male, 2=Female.
    """
    plt.style.use("ggplot")

    # Filter to cases where outcome == 1
    cases = df[df[outcome_col] == 1].copy()

    # Count cases by sex (1=Male, 2=Female), ensure both present
    by_sex = cases["SEX"].value_counts().reindex([1, 2]).fillna(0).astype(int)

    # Bar chart
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = by_sex.index.map({1: "Male", 2: "Female"})
    bars = ax.bar(labels, by_sex.values, color=["paleturquoise", "pink"])

    # Add counts on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"n = {int(height)}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Sex")
    ax.set_ylabel(f"Number of {outcome_col} = 1")
    fig.tight_layout()

    # Build summary table: totals, cases, no-cases, percentages, diff check
    rows = []
    sex_label_map = {1: "Male", 2: "Female"}
    for sex_code in [1, 2]:
        label = sex_label_map.get(sex_code, str(sex_code))

        total = (df["SEX"] == sex_code).sum()
        cases_sex = ((df["SEX"] == sex_code) & (df[outcome_col] == 1)).sum()
        no_cases_sex = ((df["SEX"] == sex_code) & (df[outcome_col] == 0)).sum()
        diff = total - (cases_sex + no_cases_sex)
        pct = (cases_sex / total * 100) if total > 0 else np.nan

        rows.append(
            {
                "Sex": label,
                "Total": total,
                f"{outcome_col} = 1 (cases)": cases_sex,
                f"{outcome_col} = 0 (non-cases)": no_cases_sex,
                "Cases %": pct,
                "Check (Total - (cases + non-cases))": diff,
            }
        )

    summary_df = pd.DataFrame(rows)

    return fig, summary_df


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

#--------------------------------------------------------------------
# MODELS
#--------------------------------------------------------------------
def get_models(random_state=42):
    return {
        "LogReg (balanced)": LogisticRegression(
            solver="liblinear",  #We used the liblinear solver because it is well suited for binary logistic regression with moderate sample sizes and class imbalance.
            class_weight="balanced",
            max_iter=1000,
        ),
        "Random Forest (balanced)": RandomForestClassifier(
            n_estimators=300,
            random_state=random_state,
            class_weight="balanced",
        ),
        "SVM (balanced)": SVC(
            kernel="rbf",
            class_weight="balanced",
            probability=True,   # needed for ROC/AUC
            random_state=random_state,
        ),
        "Neural Net (MLP)": MLPClassifier(
            hidden_layer_sizes=(32, 16),   
            activation="relu",
            solver="adam",
            alpha=0.001,                  
            max_iter=200,
            early_stopping=True,           # stops if validation score stops improving
            random_state=random_state,
        ),            
    }


def run_cross_validation(models, X, y, n_splits=5, random_state=42):
    """
    Returns a dataframe of mean CV scores for each model.
    Uses stratified k-fold and multiple metrics like in Colab.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    rows = []
    for name, model in models.items():
        scores = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        rows.append(
            {
                "Model": name,
                "CV Accuracy (mean)": np.mean(scores["test_accuracy"]),
                "CV Precision (mean)": np.mean(scores["test_precision"]),
                "CV Recall (mean)": np.mean(scores["test_recall"]),
                "CV F1 (mean)": np.mean(scores["test_f1"]),
                "CV ROC AUC (mean)": np.mean(scores["test_roc_auc"]),
            }
        )

    return pd.DataFrame(rows).sort_values(by="CV ROC AUC (mean)", ascending=False)


def fit_models_and_eval(models, X_train_scaled, X_test_scaled, y_train, y_test):
    """
    Fits each model on train, evaluates on test.
    Returns:
      results_df, confusion_matrices_dict, classification_reports_dict, roc_data_dict
    """
    results = []
    conf_mats = {}
    reports = {}
    roc_data = {}  # name -> (fpr, tpr, auc)

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        test_acc = accuracy_score(y_test, y_pred)

        conf_mats[name] = confusion_matrix(y_test, y_pred)
        reports[name] = classification_report(y_test, y_pred, output_dict=False)

        # ROC (need probabilities)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            roc_data[name] = (fpr, tpr, roc_auc)

        results.append({"Model": name, "Test Accuracy": test_acc})

    results_df = pd.DataFrame(results).sort_values(by="Test Accuracy", ascending=False)
    return results_df, conf_mats, reports, roc_data

def plot_confusion_matrix_heatmap(cm, title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        linewidths=0.5,
        linecolor="white",
        xticklabels=["Pred 0", "Pred 1"],
        yticklabels=["Actual 0", "Actual 1"],
        ax=ax,
    )
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    fig.tight_layout()
    return fig

def plot_roc_comparison(roc_data, title):
    """Overlay ROC curves for multiple models."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, (fpr, tpr, roc_auc) in roc_data.items():
        ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    fig.tight_layout()
    return fig

def predict_with_threshold(model, X, threshold=0.5):
    """
    Returns y_prob (P(class=1)) and y_pred based on a custom threshold.
    Requires model to support predict_proba.
    """
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    return y_prob, y_pred



# -------------------------------------------------------------------
# MAIN APP
# -------------------------------------------------------------------

def main():
    st.title("Framingham Heart Study: CVD Risk Analysis by Sex")
    st.write("Interactive report on the loading, cleaning and analysis of the Framingham dataset")

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
            "Research question",
            "Raw data overview",
            "Risk profile by sex",
            "Outcomes by sex",
            "Consistency checks",
            "Missing data",
            "Winsorization summary",
            "Modeling: CVD",
            "Modeling: DEATH",
            "Conclusion",
        ],
    )

    #----------------------------------------------------------------
    # RESEARCH QUESTION
    #----------------------------------------------------------------
    if view == "Research question":
        st.subheader("Research Question")
        st.markdown("PICOT formatted research question:")
        st.info("In adults in the Framingham Heart Study (P), "\
            "how well do baseline cardiovascular risk factors, including sex (I), " \
            "predict incident cardiovascular events and all-cause mortality over 24 years (O, T), "
            "and does baseline risk differ between men and women (C)? ")
        
        st.subheader("Aim")
        st.markdown("""
                    **The aims of this analysis are to:**
                    1. Describe gender differences in risk factors
                    2. Build predictive models for selected outcomes
                    """)

    
    # ----------------------------------------------------------------
    # RAW DATA OVERVIEW
    # ----------------------------------------------------------------
    elif view == "Raw data overview":
        st.subheader("Raw Data: Period 1")
        st.info("""
                We restrict all analyses to **Period 1**.

        We were given a dataset spans examinations conducted between 1956 to 1968. These examinations happened in exam cycles of the **Original Framingham Heart Study cohort**.
        Therefore, this data does not capture the true cohort baseline established at the first Framingham examination in 1948. 
        As a result, some participants may already have experienced cardiovascular events prior to "our" Period 1 examination.
        To ensure a valid and interpretable baseline for our research question, we treat Period 1 as the analytic baseline and apply additional exclusion criteria when examining incident cardiovascular outcomes 
        (e.g., excluding participants with prevalent disease at Period 1).

        Including data from subsequent exam periods (Periods 2 and 3) would introduce several issues, including:
                
        - Overlapping participants with different baseline times,
        - Inconsistent availability of key risk factor measurements across exam cycles, 
        - Inconsistencies in defining the population “at risk” for incident outcomes.

        Restricting the analysis to Period 1 ensures that all participants:
        - Are observed at a common baseline time point,
        - Have uniformly recorded baseline risk factors,
        - Are followed prospectively for outcomes using the same time origin.

        **Benefits:**
        - Clearly defined and consistent baseline for risk factor assessment
        - Uniform measurement availability across participants
        - Valid estimation of incident cardiovascular outcomes

        **Limitations:**
        - Reduces sample size, potentially lowering statistical power and increasing variance
        - Limits generalizability to modern populations, as the cohort reflects mid-20th-century risk profiles and clinical practices        
       """)

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
    # RISK PROFILE BY SEX: 
    # ----------------------------------------------------------------
    elif view == "Risk profile by sex":
        st.subheader(f"Risk Profile by Sex: Period 1")

        risk_df = make_risk_profile_df(period1_df)

        st.write("**Selected risk profile columns:**")
        st.write(RISK_PROFILE_COLUMNS)
        st.info("These variables were selected based on their established relevance to cardiovascular disease risk and availability in Period 1."
        " Only variables measures simultaneously at baseline were included, as this ensures an ordered relationship between predictors and outcomes."
        )

        if risk_df.empty:
            st.warning("No risk profile columns found for this period.")
        else:
            # Choose a variable- avoids table with 120 columns
            variable = st.selectbox(
                "Select a variable to view summary statistics:",
                RISK_PROFILE_COLUMNS
            )

            st.subheader(f"Summary Statistics for {variable} by Sex")

            if variable not in risk_df.columns:
                st.warning(f"{variable} is not in the dataframe.")
            else:
                # Group by sex for selected variable
                grouped = risk_df.groupby("SEX")[variable]

                # Compute descriptive statistics
                stats = grouped.describe().T

                # Map SEX codes to readable column names
                stats.columns = [
                    "Male" if col == 1 else "Female" if col == 2 else str(col)
                    for col in stats.columns
                ]

                # Optional: nicer index name
                stats.index.name = "Statistic"

                st.dataframe(stats)

            # (Optional) keep this if you still want to see some raw rows
            with st.expander("Show first rows grouped by sex"):
                st.dataframe(
                    risk_df.groupby("SEX").head().sort_values(by="SEX")
                )


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

        st.subheader("Discussion")
        st.info("""
                Here we looked at the risk profile distributions for males and females as a part of EDA. At this stage, we noticed some things:
                - Box plots show  good amount of outliers, we will address this later 
                - When checking the CIGPDAY feature, there was a peak for n=20. This is the number of cigarettes in a pack, and could reflect some rounding down / up which could introduce bias 
                - More females were being treated with BPMEDS than men. This reflects baseline differences in treatment (rather than risk), a potential confounding factor.
                - In all previous incidences, males had more incidences than women for all
                """)

    # ----------------------------------------------------------------
    # OUTCOMES BY SEX: 
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

    # ----------Bar Plots------------------------------------------------------
        st.subheader("Visualization of outcomes")

        outcomes_df = make_outcomes_df(period1_df)

        st.write("**Outcome columns used:**")
        st.write(OUTCOME_COLUMNS)

        if outcomes_df.empty:
            st.warning("No outcome columns found for this period.")
        else:
            # --- CVD ---
            if "CVD" in outcomes_df.columns:
                st.markdown("### CVD cases by sex")

                fig_cvd, summary_cvd = plot_outcome_cases_by_sex(
                    outcomes_df,
                    outcome_col="CVD",
                    title="CVD Cases by Sex (Period 1)",
                )
                st.pyplot(fig_cvd)

                st.write("**CVD counts & percentages by sex:**")
                st.dataframe(summary_cvd)
            else:
                st.info("CVD column not found in the dataset.")

            st.markdown("---")

            # --- DEATH ---
            if "DEATH" in outcomes_df.columns:
                st.markdown("### Deaths by sex")

                fig_death, summary_death = plot_outcome_cases_by_sex(
                    outcomes_df,
                    outcome_col="DEATH",
                    title="Deaths by Sex (Period 1)",
                )
                st.pyplot(fig_death)

                st.write("**Death counts & percentages by sex:**")
                st.dataframe(summary_death)
            else:
                st.info("DEATH column not found in the dataset.")
        

        #-----------------Outcome event rates--------------------------------
        st.subheader("Overall Outcome Event Counts and Percentages")

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

        st.subheader("Discussion")
        st.info("""
                These distributions differ clearly from what we know today:
                - The sex gap is much larger (around a 16% difference) than the one observed today.
                - Nowadays most modern studies show an increased incidence of CVD and CVD-related deaths than men - the opposite is shown by the data

                We believe this difference reflects historical context:
                - CVD diagnosis was based on male expression of the diseases, leading to women being under-diagnosed in that time. This explains women seeming more protected in our data.
                - CVD-related deaths are less prevalent nowadays due to modern medicine being much more effective, as well as the recognition of sex differences in disease
                """)


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
        st.markdown("---")
        st.subheader("PREV* consistency checks (baseline medical history)")
        st.info("""
                We performed a coherence check on the binary variables PrevCHD, PrevMI, and PrevAP.
                PrevCHD was required to be consistent with PrevMI and PrevAP (since it is a combined variable for those 2).
                If either PrevMI or PrevAP equaled 1, then PrevCHD had to equal 1.
                
                Our initial reasoning was that, conversely
                PrevCHD could only equal 0 when both PrevMI and PrevAP equaled 0. When we found some inconsistencies, we re-examined this premise.
                Since coronary heart disease is am umbrella term for multiple pathologies, it is possible that these inconsistencies stem from a "different" coronary heart disease.
                Alternatively, inconsistencies in recording practices for this variable could account for the inconsistency. This is why we decided to keep the inconsistent rows.
                """)
                 
        prev_checks = prev_consistency_checks(period1_df)

        for check_name, bad_rows in prev_checks.items():
            st.markdown(f"### {check_name}")
            if bad_rows.empty:
                st.success("No inconsistencies found.")
            else:
                st.error(f"Found {len(bad_rows)} inconsistent/suspicious rows.")
                st.dataframe(bad_rows.head(20))


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

        st.subheader("Dealing with the missing values")
        st.info("""
                The features that had missing values never had a missingness percentage higher than 5% so we dropped these. 
                Our dataset is small to begin with, so data imputation would not be smart because it risks introducing data leakage.
                """)

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
        phys_cols_present = [c for c in PHYSIOLOGICAL_LIMITS.keys() if c in dropped_df.columns]
        
        before_stats = dropped_df[list(PHYSIOLOGICAL_LIMITS.keys())].describe()

        winsorized_df = winsorize_period1(dropped_df)
        after_stats = winsorized_df[list(PHYSIOLOGICAL_LIMITS.keys())].describe()

        st.markdown("### Descriptive Statistics Before Winsorization")
        st.dataframe(before_stats)

        st.info("""
                We addressed outliers by selecting physiological limits and winsorising at these values for numerical risk factors. 
                These were the ranges we accepted, anything above or below was winsorised:
                
                - Age: 18 - 110 years
                
                - Cigpday: < 80 (80 would represent chain smoking)
                
                - BMI: 10 - 70 kg/m^2
                
                - SYSBP: 60 - 300 mmHg
                
                - DIASBP: 30 - 150 mmHg
                
                - HR: 30 - (220 - 32)bpm (220bpm minus the youngest age)
                
                - TOTCHOL: 70 - 600 mg/dL (600mg/dL represents extreme familial hypercholesteremia)
                
                Then we visualised our descriptive statistics and distributions following outlier handling.
                """)



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

    #-----------------------------------------------------------------
    # VISUALIZATION AFTER WINSORIZING
    #-----------------------------------------------------------------

    #-----------------Check for remaining missings----------------------
        st.markdown("### Check for Remaining Missing Values (After Dropping + Winsorization)")

        missing_after = compute_missing_info(winsorized_df[phys_cols_present])

        if missing_after.empty:
            st.success("No missing values remain in the physiological variables after dropping and winsorization.")
        else:
            st.error("Some missing values are still present after cleaning:")
            st.dataframe(missing_after)

    #-----------------Distribution check before vs after------------------------
        st.markdown("### Distributions Before vs After Winsorization")

        if not phys_cols_present:
            st.warning("No physiological variables found for winsorization.")
        else:
            col_choice = st.selectbox(
                "Select physiological variable to compare:",
                phys_cols_present,
            )

            # Histograms before vs after
            fig_hist, axes = plt.subplots(1, 2, figsize=(10, 4))

            sns.histplot(
                dropped_df[col_choice].dropna(),
                kde=True,
                bins=30,
                ax=axes[0],
                color="grey",
            )
            axes[0].set_title(f"{col_choice} – Before winsorization")
            axes[0].set_xlabel(col_choice)
            axes[0].set_ylabel("Count")

            sns.histplot(
                winsorized_df[col_choice].dropna(),
                kde=True,
                bins=30,
                ax=axes[1],
                color="green",
            )
            axes[1].set_title(f"{col_choice} – After winsorization")
            axes[1].set_xlabel(col_choice)
            axes[1].set_ylabel("Count")

            fig_hist.tight_layout()
            st.pyplot(fig_hist)

            # Boxplots before vs after
            fig_box, axes_box = plt.subplots(1, 2, figsize=(10, 3))

            sns.boxplot(
                x=dropped_df[col_choice].dropna(),
                ax=axes_box[0],
                color="grey",
            )
            axes_box[0].set_title(f"{col_choice} – Before winsorization")
            axes_box[0].set_xlabel(col_choice)

            sns.boxplot(
                x=winsorized_df[col_choice].dropna(),
                ax=axes_box[1],
                color="green",
            )
            axes_box[1].set_title(f"{col_choice} – After winsorization")
            axes_box[1].set_xlabel(col_choice)

            fig_box.tight_layout()
            st.pyplot(fig_box)    

    # ----------------------------------------------------------------
    # MODELING: CVD
    # ----------------------------------------------------------------
    elif view == "Modeling: CVD":
        st.subheader("Modeling: Incident CVD (Period 1)")

        st.markdown("### 1. Analytic dataset overview")
        st.write("Shape of incident CVD dataset (rows, columns):", incident_cvd_df.shape)

        st.info("""
        Our introduction (see raw data overview), touches on the reasoning behind the exclusion of previous disease incidence (PREVCHD, PREVSTRK) when examining CVD. 
        This exclusion ensures that all individuals included in the analysis were genuinely at risk of experiencing a first CVD event at the start of follow-up.
                
        Including participants with pre-existing CVD would conflate disease prevalence with incidence, as such individuals are no longer susceptible to a first occurrence of the 
        outcome and instead represent a population at risk for recurrence.
        This would bias estimates of incident risk and obscure associations between baseline risk factors and new-onset disease.
                
        Restricting the analytic cohort to participants free of CVD at baseline aligns with standard epidemiologic practice and allows for valid estimation of incident CVD risk using baseline predictors.
                """)
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

        st.markdown("### 4. Baseline Logistic regression model CVD (Unbalanced)")
        st.caption( "This baseline model uses standard logistic regression without class balancing," \
        "serving as a reference point for comparison with more advanced models.")

        # Fit logistic regression on standardized predictors
        model_cvd = LogisticRegression(solver="liblinear", max_iter = 1000)
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
  
        st.markdown("**Confusion matrix (test set):**")
        fig_cm = plot_confusion_matrix_heatmap(cm, title="Confusion Matrix")
        st.pyplot(fig_cm)


        # Classification report
        st.markdown("**Classification report (test set):**")
        report_dict = classification_report(y_test, y_test_pred, output_dict=True)
        report_df = pd.DataFrame(report_dict).T
        st.dataframe(report_df.style
                        .format("{:.2f}")
                        .background_gradient(cmap = "Blues", subset = ["precision", "recall", "f1-score"])
                        )

        st.markdown("### 5. Compare models (LogReg, Random Forest, SVM, NN)")

        models = get_models(random_state=42)

        # --- Cross-validation on TRAINING DATA ONLY (best practice) ---
        st.markdown("#### 5-fold Cross-Validation (on training set)")
        cv_summary = run_cross_validation(models, X_train_scaled, y_train, n_splits=5, random_state=42)
        st.dataframe(cv_summary)

        # --- Fit models + evaluate on test set ---
        st.markdown("#### Test Set Performance")
        results_df, conf_mats, reports, roc_data = fit_models_and_eval(
            models,
            X_train_scaled, X_test_scaled,
            y_train, y_test
        )
        st.dataframe(results_df)

        # Choose a model to inspect details
        model_choice = st.selectbox("Select model for detailed metrics:", list(models.keys()), key="cvd_model_select")

        st.markdown("**Confusion matrix (test set):**")
        cm = conf_mats[model_choice]
        fig_cm = plot_confusion_matrix_heatmap(cm, title="Confusion Matrix")
        st.pyplot(fig_cm)


        st.markdown("**Classification report (test set):**")
        report_dict = classification_report(y_test, y_test_pred, output_dict=True)
        report_df = pd.DataFrame(report_dict).T
        st.dataframe(report_df.style
                        .format("{:.2f}")
                        .background_gradient(cmap = "Blues", subset = ["precision", "recall", "f1-score"])
                        )

        # ROC comparison plot (only models with predict_proba)
        if roc_data:
            st.markdown("#### ROC Curves (test set)")
            fig_roc = plot_roc_comparison(roc_data, "ROC Curve Comparison – CVD")
            st.pyplot(fig_roc)
        else:
            st.info("No ROC data available (models missing predict_proba).")

            st.markdown("### Threshold tuning (test set)")

        chosen_model = models[model_choice]

        if not hasattr(chosen_model, "predict_proba"):
            st.info("This model does not support probability outputs (predict_proba). Threshold tuning unavailable.")
        else:
            threshold = st.slider(
                "Select probability threshold for class=1",
                min_value=0.05,
                max_value=0.95,
                value=0.50,
                step=0.05,
                key="cvd_threshold_slider",
            )

            y_prob, y_pred_thr = predict_with_threshold(chosen_model, X_test_scaled, threshold=threshold)

            acc_thr = accuracy_score(y_test, y_pred_thr)
            cm_thr = confusion_matrix(y_test, y_pred_thr)

            st.write(f"Accuracy at threshold {threshold:.2f}: **{acc_thr:.3f}**")

            st.markdown("**Confusion matrix (threshold tuned):**")
            st.dataframe(
                pd.DataFrame(
                    cm_thr,
                    index=["Actual 0", "Actual 1"],
                    columns=["Pred 0", "Pred 1"],
                )
            )
            fig_cm_thr = plot_confusion_matrix_heatmap(cm_thr, title="Confusion Matrix (Threshold tuned)")
            st.pyplot(fig_cm_thr)


            st.markdown("**Classification report (threshold tuned):**")
            report_dict = classification_report(y_test, y_pred_thr, output_dict=True)
            report_df = pd.DataFrame(report_dict).T
            st.dataframe(report_df.style
                        .format("{:.2f}")
                        .background_gradient(cmap = "Blues", subset = ["precision", "recall", "f1-score"])
                        )   

            st.subheader("Discussion")
            st.info("""
                    **Classical models (logistic regression, random forest, SVM):**

                    Hyperparameter tuning and decision threshold optimisation were explored but resulted in negligible changes to performance metrics. Accuracy and related measures remained largely unchanged, so further optimisation results are not reported.
                    
                    **Neural networks:**
                    
                    Neural network models performed poorly and showed high variability across runs, likely due to the small dataset size and the data-intensive nature of these models.
                    
                    **Model stability:**
                    Neural network outcomes fluctuated substantially between extreme and more plausible predictions, indicating sensitivity to random initialisation and limited robustness.
                    """)      

    # ----------------------------------------------------------------
    # MODELING: DEATH
    # ----------------------------------------------------------------
    elif view == "Modeling: DEATH":
        st.subheader("Modeling: All-cause Mortality (Period 1)")

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
        fig_corr_d = plot_corr_heatmap(X_death, "Correlation Heatmap: DEATH Predictors")
        st.pyplot(fig_corr_d)

        st.markdown("### 4. Baseline Logistic regression model DEATH (Unbalanced)")

        model_death = LogisticRegression(solver="liblinear")
        model_death.fit(X_train_scaled_d, y_train_d)

        y_train_pred_d = model_death.predict(X_train_scaled_d)
        y_test_pred_d = model_death.predict(X_test_scaled_d)

        train_acc_d = accuracy_score(y_train_d, y_train_pred_d)
        test_acc_d = accuracy_score(y_test_d, y_test_pred_d)

        st.write(f"Training accuracy: **{train_acc_d:.3f}**")
        st.write(f"Test accuracy: **{test_acc_d:.3f}**")

        st.markdown("**Confusion matrix (test set):**")
        
        cm_d = confusion_matrix(y_test_d, y_test_pred_d)
        fig_cm_d_df = plot_confusion_matrix_heatmap(cm_d, title="Confusion Matrix")
        st.pyplot(fig_cm_d_df)

        st.markdown("**Classification report (test set):**")            
        report_d = classification_report(y_test_d, y_test_pred_d, output_dict=True)
        report_df = pd.DataFrame(report_d).T
        st.dataframe(report_df.style
                        .format("{:.2f}")
                        .background_gradient(cmap = "Blues", subset = ["precision", "recall", "f1-score"])
                        )     
        
        st.markdown("### 5. Compare models (LogReg, Random Forest, SVM, NN)")

        models = get_models(random_state=42)

        st.markdown("#### 5-fold Cross-Validation (on training set)")
        cv_summary = run_cross_validation(models, X_train_scaled_d, y_train_d, n_splits=5, random_state=42)
        st.dataframe(cv_summary)

        st.markdown("#### Test Set Performance")
        results_df, conf_mats, reports, roc_data = fit_models_and_eval(
            models,
            X_train_scaled_d, X_test_scaled_d,
            y_train_d, y_test_d
        )
        st.dataframe(results_df)

        model_choice = st.selectbox("Select model for detailed metrics:", list(models.keys()), key="death_model_select")

        st.markdown("**Confusion matrix (test set):**")
        cm = conf_mats[model_choice]

        fig_cm = plot_confusion_matrix_heatmap(cm, title="Confusion Matrix")
        st.pyplot(fig_cm)

        st.markdown("**Classification report (test set):**")
        st.dataframe(report_df.style
                        .format("{:.2f}")
                        .background_gradient(cmap = "Blues", subset = ["precision", "recall", "f1-score"])
                        )

        if roc_data:
            st.markdown("#### ROC Curves (test set)")
            fig_roc = plot_roc_comparison(roc_data, "ROC Curve Comparison – DEATH")
            st.pyplot(fig_roc)
        else:
            st.info("No ROC data available (models missing predict_proba).")

        st.markdown("### Threshold tuning (test set)")

        chosen_model = models[model_choice]

        if not hasattr(chosen_model, "predict_proba"):
            st.info("This model does not support probability outputs (predict_proba). Threshold tuning unavailable.")
        else:
            threshold = st.slider(
                "Select probability threshold for class=1",
                min_value=0.05,
                max_value=0.95,
                value=0.50,
                step=0.05,
                key="death_threshold_slider",
            )

            y_prob, y_pred_thr = predict_with_threshold(chosen_model, X_test_scaled_d, threshold=threshold)

            acc_thr = accuracy_score(y_test_d, y_pred_thr)
            cm_thr = confusion_matrix(y_test_d, y_pred_thr)

            st.write(f"Accuracy at threshold {threshold:.2f}: **{acc_thr:.3f}**")

            st.markdown("**Confusion matrix (threshold tuned):**")
            fig_cm_thr = plot_confusion_matrix_heatmap(cm_thr, title="Confusion Matrix")
            st.pyplot(fig_cm_thr)

            st.markdown("**Classification report (threshold tuned):**")
            report_d = classification_report(y_test_d, y_pred_thr, output_dict=True)    
            report_df = pd.DataFrame(report_d).T
            st.dataframe(report_df.style
                        .format("{:.2f}")
                        .background_gradient(cmap = "Blues", subset = ["precision", "recall", "f1-score"])
                        )  

            st.subheader("Discussion")
            st.info("""
                    **Classical models (logistic regression, random forest, SVM):**

                    Hyperparameter tuning and decision threshold optimisation were explored but resulted in negligible changes to performance metrics. Accuracy and related measures remained largely unchanged, so further optimisation results are not reported.
                    
                    **Neural networks:**
                    
                    Neural network models performed poorly and showed high variability across runs, likely due to the small dataset size and the data-intensive nature of these models.
                    
                    **Model stability:**
                    Neural network outcomes fluctuated substantially between extreme and more plausible predictions, indicating sensitivity to random initialisation and limited robustness.
                    """)       
    #--------------------------------------------------------------------------------------------------------------
    # CONCLUSION
    #---------------------------------------------------------------------------------------------------------------
        elif view == "Conclusion":
        st.subheader("Conclusion")
        st.write("""
                 Our RQ: “In adults in the Framingham Heart Study (P), how well do baseline cardiovascular risk factors, 
                 including sex (I), predict incident cardiovascular events and all-cause mortality over 24 years (O, T), 
                 and does baseline risk differ between men and women (C)?”
                
                 To answer this question, we began by performing exploratory data analysis, visualising and using descriptive 
                 statistics to observe the risk profiles and outcomes by sex in Period 1.
                 We then selected features relevant to our RQ (risk factors and outcomes CVD as well as death).
                 We then created incident datarames for our outcomes (CVD and death). For the CVD dataframe, we filtered out 
                 previous incidents of CVD using PREV* variables. The dataframe for Death did not need this exclusion. 
                 Total 2 data frames, 1 for each outcome: this prevents data leakage as previous incidences are very highly 
                 correlated with their respective outcomes. 
                
                 Next, we cleaned the data, this involved:
                - Identifying missing values and dropping them (small proportion)
                - Handling outliers by applying physiological thresholds and winsorising extreme outliers.
                - Following this, we checked for erroneous data/inconsistencies between variables like CURSMOKE and CIGPDAY, as well as all PREV- variables. 
                We plotted our data again to observe changes before and after our cleaning.
                 
                 Before Standardisation, we performed the train/test split for both our CVD and Death df.
                 The continuous variables listed were standardized using z-score normalization.
                 
                Our data was then ready to train 3 different ML models:
                1. Logistic regression (CVD and Death separately)
                2. Random forest (CVD and Death separately)
                3. Support Vector Machine (CVD and Death separately)
                (We attempted to train neural networks but our dataset was too small to give good results)

                 Finally, we applied k-fold analysis to all our models (6 in total) to assess performance more robustly.

                Our best model was _______, which could be explained by______, however the models did not perform well enough to say that one of them was particularly well performing. We indeed saw a difference in risk factors and outcomes by sex, however historical contexts outline how these differences are not consistent with modern knowledge.
                 """)

        st.subheader("Disclaimer")
        st.write("""
                The streamlit code was generated from our Colab notebook with the help of GPT-5. Some checks and pieces of code were added using AI to resolve errors that arose in the transfer process and streamline the report.
                """)

if __name__ == "__main__":
    main()