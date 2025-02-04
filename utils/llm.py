import os
from openai import OpenAI

def get_cleaning_suggestions(data_summary, ml_task):
    """
    Use OpenAI's API to get cleaning suggestions based on the data summary.
    Ensure that the environment variable OPENAI_API_KEY is set.
    """
    openai_api_key = os.get_env("OPEN_AI_KEY")    
    if not openai_api_key:
        return "OpenAI API key not set. Please set the OPENAI_API_KEY environment variable."
    
    client = OpenAI(api_key=openai_api_key)
    prompt = (
        "Given the following data summary:\n\n"                                 
        f"{data_summary}\n\n"
        "What data cleaning steps would you recommend to prepare the data for a machine learning task?"
        "And given that the selected ML task is: " + ml_task + "\n\n"
    )
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.7,
        )
        suggestion = response.choices[0].message.content.strip()
        return suggestion
    except Exception as e:
        return f"Error fetching LLM suggestions: {e}"

def get_additional_checks(data_summary, ml_task):
    """
    Use OpenAI's API to get additional visualization/check code suggestions based on the data summary and ML task.
    The prompt instructs the LLM to provide complete, runnable Python code that defines a function named 'additional_checks(df)'.
    """
    openai_api_key = os.get_env("OPEN_AI_KEY")  
    if not openai_api_key:
        return "OpenAI API key not set. Please set the OPENAI_API_KEY environment variable."
    
    client = OpenAI(api_key=openai_api_key)
    prompt = (
        "Given the following data summary:\n\n"
        f"{data_summary}\n\n"
        "And given that the selected ML task is: " + ml_task + "\n\n"
        "Please suggest additional visualizations or checks that should be performed to ensure the dataset is ready for the ML task. "
        "Provide complete, runnable Python code that defines a function named 'additional_checks(df)' which, when executed, produces these visualizations or checks. "
        "Include necessary imports (e.g., matplotlib and seaborn) and ensure the code is self-contained."
    )
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.7,
        )
        code = response.choices[0].message.content.strip()
        return code
    except Exception as e:
        return f"Error fetching additional checks: {e}"
