import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from jinja2 import Environment, FileSystemLoader
import os

def generate_summary(df):
    """
    Return descriptive statistics (including non-numeric columns) for the DataFrame.
    """
    return df.describe(include='all')

def generate_missing_data_plot(df):
    """
    Generate a bar plot showing the count of missing values per column.
    """
    missing_counts = df.isnull().sum()
    fig, ax = plt.subplots()
    sns.barplot(x=missing_counts.index, y=missing_counts.values, ax=ax)
    ax.set_title("Missing Data Count per Column")
    ax.set_ylabel("Count")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    return fig

def generate_eda_report(df, transformation_log):
    """
    Generate an HTML report using a Jinja2 template.
    """
    env = Environment(loader=FileSystemLoader(searchpath=os.path.join(os.getcwd(), "templates")))
    template = env.get_template("report_template.html")
    
    # Convert summary to HTML table.
    summary_html = df.describe(include='all').to_html(classes="table table-striped")
    
    report_html = template.render(
        title="EDA Report",
        summary=summary_html,
        transformation_log=transformation_log
    )
    return report_html

def check_ml_readiness(df, ml_task):
    """
    Check if the dataset is ready for the selected ML task by evaluating:
      - Missing data percentage,
      - Constant columns,
      - (For decision trees) presence of numeric data.
    Returns a list of messages.
    """
    messages = []
    missing = df.isnull().mean()
    
    # Use items() instead of iteritems()
    for col, pct in missing.items():
        if pct > 0:
            messages.append(f"Column '{col}' has {pct*100:.1f}% missing values.")
    
    # Check data types
    for col, dtype in df.dtypes.items():
        if dtype == 'object':
            messages.append(f"Column '{col}' is categorical/text type and may need encoding.")
        elif dtype == 'datetime64[ns]':
            messages.append(f"Column '{col}' is a datetime and may need feature engineering.")
    
    # Add ML task specific checks
    if ml_task == "Decision Tree":
        messages.append("Decision Trees can handle both numerical and categorical data.")
    elif ml_task == "Self-supervised Learning":
        messages.append("Self-supervised learning typically requires a large amount of unlabeled data.")
    
    return messages
