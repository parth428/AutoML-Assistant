# Parth Patel
# Homework 1

from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Page setup
st.set_page_config(page_title="AutoML Assistant", page_icon="🤖", layout="wide")
st.title("🤖 AutoML Assistant")
st.caption("Step 1 → Upload & select target.  Step 2 → Run 3 models and compare results.")

# Sidebar helper
with st.sidebar:
    st.header("How to use AutoML Assistant")
    st.markdown(
        """
        1) **Upload** a CSV file  
        2) **Preview** & validate your data  
        3) **Pick a target** column  
        4) Click **Run Models** to compare 3 algorithms
        """
    )
    st.markdown("---")
    st.markdown("**Tips**")
    st.markdown(
        """
        • Keep headers on the first row  
        • Use plain CSV (UTF-8)  
        • Avoid leaking the target into features
        """
    )

    st.markdown("---")
    st.markdown("**Developed by Parth Patel**")

# Helpers
def read_csv_safely(file) -> pd.DataFrame:
    for enc in ("utf-8", "latin-1"):
        try:
            file.seek(0)
            return pd.read_csv(file, encoding=enc)
        except Exception:
            continue
    file.seek(0)
    return pd.read_csv(file)

def infer_column_roles(df: pd.DataFrame, max_cat_unique: int = 20) -> Dict[str, str]:
    roles: Dict[str, str] = {}
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_datetime64_any_dtype(s):
            roles[col] = "datetime"
        elif pd.api.types.is_numeric_dtype(s):
            roles[col] = "categorical (low-cardinality numeric)" if s.nunique(dropna=True) <= max_cat_unique else "numeric"
        else:
            roles[col] = "categorical"
    return roles

def detect_task_type(df: pd.DataFrame, target: str) -> str:
    s = df[target]
    if pd.api.types.is_numeric_dtype(s) and s.nunique(dropna=True) > 20:
        return "regression"
    return "classification"

def split_feature_types(df: pd.DataFrame, target: str) -> Tuple[List[str], List[str]]:
    X = df.drop(columns=[target])
    cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    return num_cols, cat_cols

def make_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    num_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")),
                               ("scaler", StandardScaler())])
    cat_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")),
                               ("oh", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer(
        transformers=[("num", num_pipe, num_cols),
                      ("cat", cat_pipe, cat_cols)]
    )

# Step 1 Upload & Preview
uploaded = st.file_uploader("📤 Upload CSV", type=["csv"])

if "df" not in st.session_state:
    st.session_state.df = None
    st.session_state.target = None
    st.session_state.task_type = None

if uploaded:
    try:
        st.session_state.df = read_csv_safely(uploaded)
    except Exception as e:
        st.error(f"Could not read file: {e}")
        st.stop()

if st.session_state.df is None:
    st.info("Upload a CSV to begin. Or click the button to load a small sample dataset.")
    if st.button("Load Sample Dataset (Synthetic Classification)"):
        rng = np.random.default_rng(42)
        n = 250
        x1 = rng.normal(0, 1, n)
        x2 = rng.normal(5, 2, n)
        x3 = rng.integers(0, 5, n)
        y = (x1 * 0.8 + x2 * -0.3 + (x3 == 2) * 1.2 + rng.normal(0, 0.7, n) > 0).astype(int)
        demo = pd.DataFrame({"feature_a": x1, "feature_b": x2, "feature_c": x3, "target": y})
        st.session_state.df = demo
        st.success("Sample dataset loaded.")
else:
    st.success("Upload successful. Preview below.")

df: Optional[pd.DataFrame] = st.session_state.df

if df is not None:
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        st.subheader("🔎 Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
    with c2:
        st.metric("Rows", len(df))
        st.metric("Columns", df.shape[1])
    with c3:
        st.metric("Missing cells", int(df.isna().sum().sum()))
        st.metric("Duplicate rows", int(df.duplicated().sum()))

    roles = infer_column_roles(df)
    st.subheader("🧭 Column Roles (inferred)")
    st.caption("Quick heuristic for guidance—no need to change your data here.")
    st.dataframe(pd.DataFrame({"column": list(roles.keys()), "role": list(roles.values())}),
                 use_container_width=True)

    st.subheader("🎯 Choose Target Column")
    default_idx = list(df.columns).index("target") if "target" in df.columns else 0
    target = st.selectbox("Target", options=df.columns, index=default_idx)
    st.session_state.target = target
    task_type = detect_task_type(df, target)
    st.session_state.task_type = task_type
    st.caption(f"Detected task type: **{task_type.capitalize()}**")

# Step 2 — Run 3 Models
    st.markdown("---")
    st.subheader("⚙️ Model Processing")
    run_now = st.button("🚀 Run 3 Models")

    if run_now:
        # Prepare data
        X = df.drop(columns=[target]).copy()
        y = df[target].copy()

        # Drop datetime columns
        for col in X.columns:
            if pd.api.types.is_datetime64_any_dtype(X[col]):
                X = X.drop(columns=[col])

        num_cols, cat_cols = split_feature_types(df, target)
        preprocessor = make_preprocessor(num_cols=[c for c in num_cols if c in X.columns],
                                         cat_cols=[c for c in cat_cols if c in X.columns])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y if task_type == "classification" else None
        )

        # Define models
        if task_type == "classification":
            model_specs = [
                ("Logistic Regression", LogisticRegression(max_iter=1000)),
                ("Random Forest", RandomForestClassifier(n_estimators=200, random_state=42)),
                ("Gradient Boosting", GradientBoostingClassifier(random_state=42)),
            ]
        else:
            model_specs = [
                ("Linear Regression", LinearRegression()),
                ("Random Forest", RandomForestRegressor(n_estimators=300, random_state=42)),
                ("Gradient Boosting", GradientBoostingRegressor(random_state=42)),
            ]

        progress = st.progress(0)
        status = st.empty()
        results = []
        preds_store = {}

        total = len(model_specs)
        for i, (name, base_model) in enumerate(model_specs, start=1):
            status.write(f"Training **{name}** …")
            pipe = Pipeline(steps=[("prep", preprocessor), ("model", base_model)])
            pipe.fit(X_train, y_train)

            # Predict and score
            y_pred = pipe.predict(X_test)

            if task_type == "classification":
                acc = float(accuracy_score(y_test, y_pred))
                f1 = float(f1_score(y_test, y_pred, average="weighted"))
                score = acc 
                results.append({"Model": name, "Accuracy": acc, "F1_weighted": f1})
            else:
                rmse = float(mean_squared_error(y_test, y_pred, squared=False))
                mae = float(mean_absolute_error(y_test, y_pred))
                r2 = float(r2_score(y_test, y_pred))
                score = r2
                results.append({"Model": name, "RMSE": rmse, "MAE": mae, "R2": r2})

            preds_store[name] = {
                "y_true": y_test.reset_index(drop=True),
                "y_pred": pd.Series(y_pred).reset_index(drop=True),
            }

            progress.progress(int(i / total * 100))

        status.write("✅ Done training all models.")
        progress.empty()

# Results dashboard
        st.subheader("📊 Results Dashboard")
        res_df = pd.DataFrame(results)

# Choose best model
        if task_type == "classification":
            best_row = res_df.sort_values("Accuracy", ascending=False).iloc[0]
            best_label = f"Best: {best_row['Model']} (Accuracy {best_row['Accuracy']:.3f})"
        else:
            best_row = res_df.sort_values("R2", ascending=False).iloc[0]
            best_label = f"Best: {best_row['Model']} (R² {best_row['R2']:.3f})"

        st.markdown(f"**{best_label}**")

        st.dataframe(res_df, use_container_width=True)

        # Bar chart of the primary metric
        st.caption("Primary metric comparison")
        if task_type == "classification":
            chart_df = res_df[["Model", "Accuracy"]].set_index("Model")
        else:
            chart_df = res_df[["Model", "R2"]].set_index("Model")
        st.bar_chart(chart_df)

        # Download predictions & metrics
        st.markdown("---")
        st.subheader("⬇️ Download Outputs")
        best_name = best_row["Model"]
        best_preds = preds_store[best_name]
        pred_out = pd.DataFrame({"y_true": best_preds["y_true"], "y_pred": best_preds["y_pred"]})
        st.download_button("Download Best Model Predictions (CSV)",
                           pred_out.to_csv(index=False).encode("utf-8"),
                           file_name="best_model_predictions.csv",
                           mime="text/csv")
        st.download_button("Download Metrics (CSV)",
                           res_df.to_csv(index=False).encode("utf-8"),
                           file_name="model_metrics.csv",
                           mime="text/csv")

        # Plain-English explanations
        st.markdown("---")
        st.subheader("🧾 Plain-English Summary")
        if task_type == "classification":
            st.write(
                "• **Accuracy** = proportion of correct predictions across all classes.  "
                "• **F1 (weighted)** balances precision/recall and weights by class size.  "
                "Higher is better for both."
            )
        else:
            st.write(
                "• **R²** explains the fraction of variance captured by the model (higher is better).  "
                "• **RMSE/MAE** measure typical prediction error in target units (lower is better)."
            )
