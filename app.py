import streamlit as st
import pandas as pd
import os
from datetime import datetime

# Import our helper modules from the utils package
from utils import eda, cleaning, llm

# Global log to track transformation steps (used for export)
transformation_log = []

def export_transformation_script(log):
    """
    Create a Python script file containing the transformation steps.
    """
    script_lines = [
        "# Transformation Script",
        "# Generated on: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "",
        "import pandas as pd",
        "",
        "def transform_data(df):",
        "    # Transformation steps applied",
    ]
    for step in log:
        # Each transformation line is indented inside the function.
        script_lines.append("    " + step)
    script_lines.append("    return df")
    script_content = "\n".join(script_lines)
    
    export_path = os.path.join("export", "transformation_script.py")
    with open(export_path, "w") as f:
        f.write(script_content)
    return export_path

def export_eda_report(df, log):
    """
    Generate an HTML EDA report using a Jinja2 template and save it.
    """
    report_html = eda.generate_eda_report(df, log)
    report_path = os.path.join("export", f"eda_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
    with open(report_path, "w") as f:
        f.write(report_html)
    return report_path

def main():
    st.title("Agentic EDA & ML Readiness Checker")
    
    # --- Sidebar Section ---
    st.sidebar.header("Input Options")
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
    ml_task = st.sidebar.selectbox("Select your ML task", ["Decision Tree", "Self-supervised Learning", "Other"])
    
    # Initialize session state for LLM code if not already present.
    if 'llm_code' not in st.session_state:
        st.session_state.llm_code = None
    
    # --- Main Section ---
    if uploaded_file is not None:
        try:
            # First read the CSV without date parsing to check columns
            df = pd.read_csv(uploaded_file, low_memory=False)
            
            # Reset the file pointer
            uploaded_file.seek(0)
            
            # Check if 'Date' column exists
            date_columns = [col for col in df.columns if 'date' in col.lower()]
            
            if date_columns:
                # Read CSV with date parsing for found date columns
                df = pd.read_csv(uploaded_file, low_memory=False, parse_dates=date_columns)
                st.info(f"Date parsing applied to columns: {', '.join(date_columns)}")
            else:
                # Read CSV without date parsing
                df = pd.read_csv(uploaded_file, low_memory=False)
                st.warning("No date columns detected in the dataset")
            
            # Convert object columns to string type, except for date columns
            for col in df.select_dtypes(['object']).columns:
                if col not in date_columns:
                    df[col] = df[col].astype(str)
            
            st.subheader("Data Preview")
            st.dataframe(df.head())
            
            st.subheader("Initial EDA Summary")
            summary = eda.generate_summary(df)
            st.write(summary)
            
            st.subheader("Missing Data Visualization")
            missing_fig = eda.generate_missing_data_plot(df)
            st.pyplot(missing_fig)
            
            st.subheader("ML Readiness Check")
            readiness_messages = eda.check_ml_readiness(df, ml_task)
            for message in readiness_messages:
                st.info(message)
            
            # --- LLM-Powered Cleaning Suggestions ---
            if st.button("Get LLM Cleaning Suggestions"):
                suggestion = llm.get_cleaning_suggestions(summary.to_string(), ml_task)
                st.subheader("LLM Cleaning Suggestions")
                st.write(suggestion)
            
            st.subheader("Data Cleaning Options")
            # Checkbox to impute missing values
            if st.checkbox("Impute Missing Values"):
                df = cleaning.impute_missing_values(df, transformation_log)
                st.success("Missing values imputed using column means.")
            # Checkbox to drop duplicate rows
            if st.checkbox("Drop Duplicate Rows"):
                df = cleaning.drop_duplicates(df, transformation_log)
                st.success("Duplicate rows dropped.")
            
            st.subheader("Cleaned Data Preview")
            st.dataframe(df.head())
            
            st.subheader("Updated EDA")
            updated_summary = eda.generate_summary(df)
            st.write(updated_summary)
            
            st.subheader("Updated Missing Data Visualization")
            updated_missing_fig = eda.generate_missing_data_plot(df)
            st.pyplot(updated_missing_fig)
            
            st.subheader("ML Readiness Check (After Cleaning)")
            updated_readiness_messages = eda.check_ml_readiness(df, ml_task)
            for message in updated_readiness_messages:
                st.info(message)
            
            # --- LLM Feedback Loop for Additional Checks ---
            st.subheader("Additional Visualizations / Checks (LLM Feedback Loop)")
            if st.button("Get Additional Checks Suggestions"):
                additional_code = llm.get_additional_checks(summary.to_string(), ml_task)
                st.session_state.llm_code = additional_code
                st.code(additional_code, language='python')
            
            if st.button("Run Additional Checks Code"):
                if st.session_state.llm_code:
                    code_str = st.session_state.llm_code
                    local_namespace = {}
                    try:
                        exec(code_str, local_namespace)
                        if 'additional_checks' in local_namespace:
                            st.info("Running additional_checks...")
                            local_namespace['additional_checks'](df)
                        else:
                            st.error("The provided code does not define a function named 'additional_checks'.")
                    except Exception as e:
                        st.error(f"Error executing the additional checks code: {e}")
                else:
                    st.error("No additional checks code available. Please click 'Get Additional Checks Suggestions' first.")
            
            # --- Export Options ---
            st.subheader("Export Options")
            if st.button("Export Cleaned Data as CSV"):
                csv_export_path = os.path.join("export", f"cleaned_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
                df.to_csv(csv_export_path, index=False)
                st.success(f"Cleaned data exported to {csv_export_path}")
            
            if st.button("Export Transformation Script"):
                script_path = export_transformation_script(transformation_log)
                st.success(f"Transformation script exported to {script_path}")
            
            if st.button("Export EDA Report (HTML)"):
                report_path = export_eda_report(df, transformation_log)
                st.success(f"EDA report exported to {report_path}")

        except Exception as e:
            st.error(f"Error reading the CSV file: {str(e)}")
            return

if __name__ == "__main__":
    # Ensure the export directory exists.
    if not os.path.exists("export"):
        os.makedirs("export")
    main()
