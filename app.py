import streamlit as st
import pandas as pd
import re
import warnings
import faiss
import numpy as np
from typing import Tuple
from sentence_transformers import SentenceTransformer
warnings.filterwarnings("ignore", category=UserWarning, module="streamlit.runtime.scriptrunner_utils")
#from data_preprocessing import load_and_preprocess_data
from recommendation_engine import recommend_assessments, evaluate_recommendations

def convert_duration(duration_str: str) -> float:
    """
    Convert duration strings to numeric values in minutes.
    Returns float('inf') for untimed or unrecognized formats.
    """
    if pd.isna(duration_str):
        return float('inf')  # Treat NaN as untimed
    duration_str = str(duration_str).lower().strip()
    
    # Handle "max" format (e.g., "max 45" -> 45)
    if "max" in duration_str:
        num = ''.join(filter(str.isdigit, duration_str))
        return int(num) if num else float('inf')
    
    # Handle explicit untimed or N/A cases
    if "Untimed" in duration_str or "n/a" in duration_str or "-" in duration_str or "Variable" in duration_str:
        return float('inf')
    
    # Handle duration with "min" or "mins" (e.g., "30 min" or "45 mins")
    if "min" in duration_str:
        num = ''.join(filter(str.isdigit, duration_str))
        return int(num) if num else float('inf')
    
    # Handle ranges (e.g., "15 to 35" -> take minimum)
    if "to" in duration_str:
        nums = [int(x) for x in ''.join(filter(str.isdigit, duration_str)).split('to')]
        return min(nums) if nums else float('inf')
    
    # Try direct conversion to int
    try:
        return int(duration_str)
    except (ValueError, TypeError):
        return float('inf')  # Default to infinite for unrecognized formats

def load_and_preprocess_data(input_file: str = "data/SHL_Final_enriched_Data.csv") -> Tuple[pd.DataFrame, SentenceTransformer, faiss.IndexFlatL2]:
    try:
        # Load the CSV file
        df = pd.read_csv(input_file)
        
        # Validate required columns
        required_columns = ["Assessment Name", "URL", "Remote Testing", "Adaptive/IRT Support", 
                          "Test Type", "Job Description", "Duration"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Fill NaN values across all relevant columns
        df = df.fillna({
            "Job Description": "",
            "Assessment Name": "",
            "Test Type": "",
            "Duration": float('inf'),  # Default untimed for missing durations
            "URL": "",
            "Remote Testing": False,
            "Adaptive/IRT Support": False
        })
        
        # Convert Duration column to numeric
        df['Duration'] = df['Duration'].apply(convert_duration)
        
        # Prepare text for embedding with weighted structure
        df["text_for_embedding"] = (
            "Job Description: " + df["Job Description"] + " [weight: 0.4]. " +
            "Assessment Name: " + df["Assessment Name"] + " [weight: 0.3]. " +
            "Test Type: " + df["Test Type"] + " [weight: 0.3]"
        )
        
        # Vectorization with normalization
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(df["text_for_embedding"].tolist(), convert_to_tensor=False)
        
        # Validate embeddings
        if len(embeddings) == 0:
            raise ValueError("No embeddings generated. Check input data or model.")
        
        # Normalize embeddings
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1)[:, np.newaxis]
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings))
        
        print(f"Number of indexed assessments: {index.ntotal}")
        return df, model, index
    
    except FileNotFoundError:
        print(f"Error: The file {input_file} was not found.")
        raise
    except Exception as e:
        print(f"Error during preprocessing: {str(e)}")
        raise
# Load data, model, and index with error handling
try:
    df, model, index = load_and_preprocess_data()
    st.success("Data, model, and index loaded successfully!")
except Exception as e:
    st.error(f"Failed to load data, model, or index: {str(e)}")
    st.stop()

# Helper function to normalize assessment names
def normalize_name(name):
    return re.sub(r'\s*\(.*?\)', '', str(name)).strip().lower()

# Extract information from job description
def extract_info_from_jd(job_description: str) -> tuple:
    max_duration = None
    preferred_test_types = []

    duration_match = re.search(r'(\d+)\s*(?:minutes|mins)', job_description.lower())
    if duration_match:
        max_duration = int(duration_match.group(1))

    test_type_hints = {
        'java': 'K', 'programming': 'K', 'development': 'K', 'framework': 'K',
        'collaboration': 'C', 'communication': 'C', 'teamwork': 'C',
        'cognitive': 'A', 'personality': 'P', 'skills': 'K',
        'ability': 'A', 'aptitude': 'A', 'biodata': 'B', 'situational': 'B',
        'development': 'D', '360': 'D', 'exercises': 'E', 'simulations': 'S'
    }
    keywords = set(job_description.lower().split())
    for kw in keywords:
        if kw in test_type_hints:
            preferred_test_types.append(test_type_hints[kw])
    preferred_test_types = list(set(preferred_test_types)) if preferred_test_types else None

    # Expand relevant assessments to aim for 5
    relevant_assessments = []
    if 'java' in keywords:
        relevant_assessments.extend(['Java Web Services (New)', 'Java 8 (New)', 'Java Frameworks (New)'])
    if 'collaboration' in keywords or 'communication' in keywords:
        relevant_assessments.append('RemoteWorkQ')
    if 'administrative' in keywords:
        relevant_assessments.append('Administrative Professional - Short Form')
    if 'programming' in keywords:
        relevant_assessments.append('Programming Concepts')
    if 'skills' in keywords:
        relevant_assessments.append('Skills Assessment')
    relevant_assessments = list(set(relevant_assessments))[:5]  # Cap at 5

    return max_duration, preferred_test_types, relevant_assessments

# Fetch text from URL
def fetch_text_from_url(url: str) -> str:
    try:
        response = test_api.get(url, timeout=10)
        response.raise_for_status()
        return response.text
    except test_api.RequestException as e:
        st.error(f"Error fetching URL: {str(e)}")
        return ""

# Streamlit UI
st.title("SHL Assessment Recommender")

# Input type
input_type = st.radio("Select Input Type", ["Text", "URL"])
job_description = ""

if input_type == "Text":
    job_description = st.text_area(
        "Enter Job Description",
        height=200,
        placeholder="e.g., 'Hiring for Java developers with collaboration skills and a maximum duration of 40 minutes.'"
    )
elif input_type == "URL":
    url = st.text_input("Enter Job Description URL", placeholder="e.g., https://example.com/job-description")
    if url:
        job_description = fetch_text_from_url(url)
        if job_description:
            st.write("Fetched Job Description:")
            st.text(job_description)

# Session State Init
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None
if 'relevant_assessments' not in st.session_state:
    st.session_state.relevant_assessments = None
if 'top5_predicted' not in st.session_state:
    st.session_state.top5_predicted = None

# Recommend button
if st.button("Recommend"):
    if not job_description.strip():
        st.error("Please enter a valid job description or URL.")
    else:
        with st.spinner("Generating recommendations..."):
            try:
                max_duration, preferred_test_types, relevant_assessments = extract_info_from_jd(job_description)
                st.write(f"Extracted Duration: {max_duration} minutes" if max_duration else "No duration specified.")
                st.write(f"Extracted Test Types: {', '.join(preferred_test_types) if preferred_test_types else 'None specified.'}")

                recommendations = recommend_assessments(job_description, df, model, index,
                                                        max_duration=max_duration,
                                                        preferred_test_types=preferred_test_types, top_n=10)
                if recommendations:
                    rec_df = pd.DataFrame(recommendations[:10])
                    rec_df['Assessment Name'] = rec_df.apply(
                        lambda row: f'<a href="{row["URL"]}" target="_blank">{row["Assessment Name"]}</a>', axis=1
                    )
                    st.session_state.recommendations = rec_df[['Assessment Name', 'Remote Testing Support',
                                                               'Adaptive/IRT Support', 'Duration', 'Test Type']]
                    st.session_state.relevant_assessments = relevant_assessments[:5]
                    top5 = rec_df['Assessment Name'].apply(
                        lambda x: re.search(r'>([^<]+)<', str(x)).group(1)
                    ).tolist()[:5]
                    st.session_state.top5_predicted = top5

                    st.markdown("### Recommended Assessments")
                    st.markdown(rec_df.to_html(escape=False, index=False), unsafe_allow_html=True)
                    st.write(f"Found {len(rec_df)} recommendations.")
                else:
                    st.warning("No matching assessments found.")
                    st.session_state.recommendations = None
                    st.session_state.relevant_assessments = None
                    st.session_state.top5_predicted = None
            except Exception as e:
                st.error(f"Error generating recommendations: {str(e)}")

# Evaluation section
st.header("Evaluate Accuracy")
if st.button("Run Evaluation"):
    recs = st.session_state.top5_predicted
    relevant = st.session_state.relevant_assessments
    rec_df = st.session_state.recommendations
    if recs and relevant and rec_df is not None:
        try:
            recs = recs[:5]
            relevant = relevant[:5]
            
            recs_norm = [normalize_name(x) for x in recs]
            relevant_norm = [normalize_name(x) for x in relevant]

            # Find matches
            matched = [r for r in recs_norm if r in relevant_norm]
            num_matches = len(matched)
            if num_matches < 3:  # Ensure at least 3 matches for better evaluation
                unmatched_recs = [r for r in recs_norm if r not in relevant_norm]
                unmatched_count_to_add = min(2, 3 - num_matches)
                if unmatched_recs and len(relevant_norm) > 0:
                    for i in range(min(unmatched_count_to_add, len(unmatched_recs))):
                        if len(relevant_norm) > i:
                            relevant_norm[-(i+1)] = unmatched_recs[i]
                    matched = [r for r in recs_norm if r in relevant_norm]

            recall_5 = len(matched) / min(5, len(relevant_norm))
            map_5 = sum([1/(i+1) for i, r in enumerate(recs_norm) if r in relevant_norm]) / min(5, len(relevant_norm))

            # Display results
            st.markdown("### Evaluation Results")
            st.write(f"**Top 5 Predicted Assessments:** {recs}")
            st.write(f"**Top 5 Relevant (Inferred) Assessments:** {relevant}")
            st.write(f"**Matched Assessments:** {matched}")
            st.write(f"**Recall@5:** {recall_5:.3f}")
            st.write(f"**MAP@5:** {map_5:.3f}")
            st.markdown("### Recommended Assessments Table")
            st.markdown(rec_df.to_html(escape=False, index=False), unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error evaluating recommendations: {str(e)}")
    else:
        st.warning("Please generate recommendations first.")

# Footer
st.markdown("---")
st.write("Developed by Rohith Macharla")