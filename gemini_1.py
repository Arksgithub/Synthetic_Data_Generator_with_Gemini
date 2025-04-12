import streamlit as st
import pandas as pd
import google.generativeai as genai
from io import StringIO
import json

# --- Gemini Setup ---
genai.configure(api_key="GEMINI_API_KEY")
model = genai.GenerativeModel("gemini-1.5-flash")

# --- Functions ---
def generate_Synthetic_data(context: str, num_rows: int, file_format: str) -> str:
    prompt = f"""
You are a synthetic data generator assistant. Generate exactly {num_rows} rows of Synthetic data in {file_format.upper()} format based on this context:

"{context}"

Format rules:
- For CSV: include a header in the first row. Use standard comma-separated format.
- For JSON: return a list of dictionaries (array of objects). Do NOT include markdown or code fences.

Only output the data — no explanations.
"""

    try:
        response = model.generate_content(prompt)
        return response.text.strip() if hasattr(response, "text") else str(response).strip()
    except Exception as e:
        return f"ERROR: {e}"

def clean_output(output: str, file_format: str) -> str:
    if "```" in output:
        output = output.split("```")[-2] if len(output.split("```")) >= 2 else output
    if file_format == "JSON":
        output = output.strip().lstrip("json").strip()
    return output.strip()

def parse_data(output: str, file_format: str) -> pd.DataFrame:
    if file_format == "CSV":
        return pd.read_csv(StringIO(output))
    elif file_format == "JSON":
        return pd.DataFrame(json.loads(output))
    else:
        raise ValueError("Unsupported file format")

def download_file(df: pd.DataFrame, file_format: str) -> bytes:
    if file_format == "CSV":
        return df.to_csv(index=False).encode("utf-8")
    elif file_format == "JSON":
        return df.to_json(orient="records", indent=2).encode("utf-8")
    return b""

# --- Streamlit UI ---
st.set_page_config(page_title="Synthetic Data Generator with Gemini", layout="centered")
st.title("Synthetic Data Generator using Gemini LLM")


context = st.text_area("Describe the context of the data to generate", placeholder="E.g., Employee database with name, age, department, salary")
file_format = st.selectbox("Select file format", ["CSV", "JSON"])
num_records = st.number_input("Number of rows to generate", min_value=1, max_value=1000, value=10, step=1)

if st.button("🚀 Generate Synthetic Data"):
    with st.spinner("Generating data using Gemini..."):
        raw_output = generate_Synthetic_data(context, num_records, file_format)
        cleaned_output = clean_output(raw_output, file_format)

        try:
            df = parse_data(cleaned_output, file_format)
            st.success("Data generated successfully!")
            st.dataframe(df)

            data_bytes = download_file(df, file_format)
            st.download_button(
                label=f"Download {file_format}",
                data=data_bytes,
                file_name=f"Synthetic_data.{file_format.lower()}",
                mime="text/csv" if file_format == "CSV" else "application/json"
            )
        except Exception as e:
            st.error(f" Failed to parse the generated data. Error: {e}")
            with st.expander("🔍 Raw Output from Gemini"):
                st.code(raw_output)
