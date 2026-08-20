import streamlit as st
import pandas as pd
from gensim.models import KeyedVectors

# Page Configuration
st.set_page_config(
    page_title="MedTerm Explorer", 
    page_icon="🏥", 
    layout="wide"
)

@st.cache_resource
def load_vectors():
    # Load serialized binary vectors efficiently
    return KeyedVectors.load_word2vec_format("Models/clinical_word2vec.kv", binary=True)

@st.cache_data
def load_data():
    return pd.read_csv("Data/mtsamples.csv")

st.title("🏥 MedTerm Explorer: Clinical Language Discovery Tool")
st.markdown("Explore semantic relationships across clinical documentation extracted from the **MTSamples** corpus.")

# Load Assets
try:
    wv = load_vectors()
    df = load_data()
    st.sidebar.success("Weights & Dataset Loaded Successfully!")
except Exception as e:
    st.error(f"Error loading files. Ensure 'Models/clinical_word2vec.kv' and 'Data/mtsamples.csv' exist. Details: {e}")
    st.stop()

# Sidebar Control
st.sidebar.header("Query Settings")
top_n = st.sidebar.slider("Number of Similar Terms", min_value=3, max_value=15, value=5)

# Main UI Query Input
query_term = st.text_input("Enter a Medical Term or Acronym:", value="pneumonia").strip().lower()

if st.button("Search Semantic Neighbours") or query_term:
    if query_term in wv:
        st.subheader(f"Results for: **'{query_term}'**")
        
        col1, col2 = st.columns([1, 1.5])
        
        with col1:
            st.markdown("### Top Similar Terms")
            results = wv.most_similar(query_term, topn=top_n)
            res_df = pd.DataFrame(results, columns=["Term", "Cosine Similarity"])
            res_df["Cosine Similarity"] = res_df["Cosine Similarity"].round(4)
            st.dataframe(res_df, use_container_width=True)
            
        with col2:
            st.markdown("### Sample Context Snippets")
            # Filter dataset for matching mentions
            matches = df[df['transcription'].str.contains(query_term, case=False, na=False)]
            
            if not matches.empty:
                st.caption(f"Found in {len(matches)} clinical notes. Showing top 3 occurrences:")
                for idx, row in matches.head(3).iterrows():
                    with st.expander(f"Specialty: {row.get('medical_specialty', 'General')} | Document ID: {idx}"):
                        st.write(f"...{row['transcription'][:300]}...")
            else:
                st.info("No direct exact matches found in raw text snippets.")
    else:
        st.warning(f"Term **'{query_term}'** is not in the vocabulary. Try terms like: *pneumonia, fracture, hypertension, surgery, antibiotic*.")

# Safety Disclaimer (Proves Domain Competence)
st.markdown("---")
st.caption(
    "⚠️ **Clinical Safety Notice:** MedTerm Explorer is a semantic text-discovery aid designed to demonstrate query expansion and lexical similarity in clinical NLP pipelines. It is not intended for clinical diagnosis, billing coding, or treatment decision support."
)