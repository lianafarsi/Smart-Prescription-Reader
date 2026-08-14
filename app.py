import streamlit as st
import pandas as pd
import re
from transformers import pipeline

# Set page configuration
st.set_page_config(page_title="Smart Prescription Reader", page_icon="💊")

# Cache the model so it only loads once!
@st.cache_resource
def load_ner_model():
    # Using your exact chosen model from HuggingFace
    return pipeline("ner", model="d4data/biomedical-ner-all", tokenizer="d4data/biomedical-ner-all", aggregation_strategy="simple")

# --- Your Custom Functions ---
def clean_medical_text(text: str) -> str:
    """Cleans raw medical text by removing HTML tags and extra whitespaces."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def parse_ner_results(extraction_results: list) -> pd.DataFrame:
    """Parses the raw NER model output into a structured DataFrame."""
    structured_rows = []
    for item in extraction_results:
        for entity in item['entities']:
            structured_rows.append({
                "Entity Text": entity.get('word').replace('##', ''),
                "Entity Type": entity.get('entity'),
                "Confidence Score": round(entity.get('score', 0.0), 4)
            })
    return pd.DataFrame(structured_rows)

def filter_and_validate_entities(df: pd.DataFrame, min_confidence: float = 0.85) -> pd.DataFrame:
    """Filters out low confidence predictions and removes exact duplicates."""
    if df.empty:
        return df
    # Step 1: Filter by confidence
    filtered_df = df[df['Confidence Score'] >= min_confidence].copy()
    # Step 2: Remove duplicates based on text and type
    if not filtered_df.empty:
        filtered_df = filtered_df.drop_duplicates(subset=['Entity Text', 'Entity Type']).reset_index(drop=True)
    return filtered_df
# ------------------------------

# App UI
st.title("Smart Medical Text Analysis System 💊")
st.write("Enter the prescription text or medical notes in the box below to extract drugs and dosages.")

user_input = st.text_area("Write the medical text here:", height=150,
                          value="Patient presents with severe headache and acute migraine. Prescribed Sumatriptan 50mg daily. History of type 2 diabetes. The patient was advised to take Metformin 500mg twice a day.")

if st.button("Analyze Text & Extract Drugs"):
    with st.spinner("Loading AI Model and analyzing text... (This might take a minute the first time)"):

        # 1. Clean the input text
        cleaned_text = clean_medical_text(user_input)

        # 2. Load model and predict
        ner_pipeline = load_ner_model()
        raw_entities = ner_pipeline(cleaned_text)

        # Format to match your parsing function structure
        extraction_results = [{"text_id": 0, "entities": raw_entities}]

        # 3. Parse and filter results
        parsed_df = parse_ner_results(extraction_results)
        final_df = filter_and_validate_entities(parsed_df, min_confidence=0.80)

        # 4. Display the results!
        st.success("Analysis Complete!")

        if not final_df.empty:
            st.markdown("### Extracted Medical Entities:")
            # Display DataFrame as an interactive table
            st.dataframe(final_df, use_container_width=True)
        else:
            st.warning("No high-confidence medical entities found in the text.")
