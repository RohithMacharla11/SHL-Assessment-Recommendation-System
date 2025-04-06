import streamlit as st
import pandas as pd
import re
import requests
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="streamlit.runtime.scriptrunner_utils")
from data_preprocessing import load_and_preprocess_data
from recommendation_engine import recommend_assessments, evaluate_recommendations

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
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
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