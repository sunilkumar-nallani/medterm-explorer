# MedTerm Explorer 🏥🔍

**Live Demo:** [Link to your Streamlit App here]

## What is this?
MedTerm Explorer is a lightweight, interactive clinical NLP dashboard. It allows users to explore semantic relationships between medical terms, symptoms, and diagnoses. 

Instead of relying on generic language models, I engineered a custom NLP pipeline to train Skip-Gram embeddings (Word2Vec) directly on 4,999 raw, unstructured clinical records. This ensures the model actually understands domain-specific medical phrasing (e.g., grouping "chest_pain" and "myocardial_infarction").

## Why I built it
In clinical settings, medical documentation is messy and highly specialized. I wanted to build a tool that proves how custom embeddings can map these specialized relationships visually, providing rapid similarity lookups with **under 10ms query latency**. 

## Tech Stack
* **Python** (Core Logic)
* **Gensim (Word2Vec)** (Custom Embeddings & Phrase Detection)
* **Streamlit** (Interactive UI & Deployment)
* **Pandas & Scikit-learn** (Data Manipulation & PCA Visualization)

## How to run it locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/sunilkumar-nallani/medterm-explorer.git](https://github.com/sunilkumar-nallani/medterm-explorer.git)
   cd medterm-explorer
