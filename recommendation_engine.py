from sentence_transformers import SentenceTransformer, util
import numpy as np
import faiss
from typing import List, Dict, Optional
import pandas as pd
from typing import Tuple

def recommend_assessments(query_text: str, df: pd.DataFrame, model: SentenceTransformer,
                          index: faiss.IndexFlatL2, max_duration: Optional[float] = None,
                          preferred_test_types: Optional[List[str]] = None, top_n: int = 10) -> List[Dict]:
    """
    Recommend SHL assessments based on query relevance, filtered by duration and test types,
    combining FAISS similarity and skill keyword scoring.
    """
    if not query_text or not isinstance(query_text, str):
        return []

    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame")

    query_keywords = set(query_text.lower().split())
    test_type_hints = {
        'java': 'K', 'programming': 'K', 'development': 'K', 'framework': 'K',
        'collaboration': 'C', 'communication': 'C', 'teamwork': 'C',
        'cognitive': 'A', 'personality': 'P', 'skills': 'K',
        'ability': 'A', 'aptitude': 'A', 'biodata': 'B', 'situational': 'B',
        'development': 'D', '360': 'D', 'exercises': 'E', 'simulations': 'S'
    }
    skill_hints = {'spring': 1.0, 'web': 0.8, 'debugging': 0.6, 'collaboration': 0.7}

    inferred_test_types = [test_type_hints.get(kw, '') for kw in query_keywords if kw in test_type_hints]
    inferred_test_types = [t for t in inferred_test_types if t] or None
    query_skills = {kw: skill_hints.get(kw, 0.0) for kw in query_keywords if kw in skill_hints}

    # Generate query embedding
    try:
        query_embedding = model.encode([query_text])[0]
    except Exception as e:
        print(f"Error generating query embedding: {e}")
        return []

    # Search top candidates from FAISS
    D, I = index.search(np.array([query_embedding]), k=top_n * 3)  # Overfetch to filter

    results = []
    for idx, score in zip(I[0], D[0]):
        if idx >= len(df):
            continue
        row = df.iloc[idx]
        duration = row.get('Duration', float('inf'))

        # Filter by duration
        if max_duration is not None and duration > max_duration:
            continue

        # Filter by preferred or inferred test types
        effective_test_types = preferred_test_types or inferred_test_types
        if effective_test_types:
            test_types = str(row.get('Test Type', '')).replace(' ', '').lower()
            if not any(t.lower() in test_types for t in effective_test_types):
                continue

        # Skill keyword match score
        skill_match_score = 0.0
        row_text = (str(row.get('Job Description', '')) + " " + str(row.get('Assessment Name', ''))).lower()
        for skill, weight in query_skills.items():
            if skill in row_text:
                skill_match_score += weight

        # Normalize FAISS distance (L2) to similarity
        faiss_sim = 1 / (1 + score)  # convert L2 to similarity

        combined_score = 0.7 * faiss_sim + 0.3 * skill_match_score

        results.append({
            'Assessment Name': row.get('Assessment Name', ''),
            'URL': row.get('URL', ''),
            'Remote Testing Support': 'Yes' if row.get('Remote Testing', False) else 'No',
            'Adaptive/IRT Support': 'Yes' if row.get('Adaptive/IRT Support', False) else 'No',
            'Test Type': row.get('Test Type', ''),
            'Duration': str(duration) if not np.isinf(duration) else 'N/A',
            '_score': combined_score
        })

    # Sort and return top_n
    results.sort(key=lambda x: x['_score'], reverse=True)
    return [{k: v for k, v in r.items() if k != '_score'} for r in results[:top_n]]


def evaluate_recommendations(queries: List[str], relevant_assessments: List[List[str]], 
                           df: pd.DataFrame, model: SentenceTransformer, 
                           index: faiss.IndexFlatL2) -> Tuple[float, float]:
    if len(queries) != len(relevant_assessments):
        raise ValueError("Number of queries must match number of relevant assessment lists.")

    mean_recall_3 = 0.0
    mean_ap_3 = 0.0
    n = len(queries)

    for query, relevant in zip(queries, relevant_assessments):
        recs = recommend_assessments(query, df, model, index, top_n=3)
        relevant_set = set(r.lower().strip() for r in relevant if r)
        rec_set = [r['Assessment Name'].lower().strip() for r in recs]

        # Recall@3
        relevant_count = len(relevant_set)
        if relevant_count > 0:
            recall_3 = len(set(rec_set) & relevant_set) / relevant_count
        else:
            recall_3 = 0.0
        mean_recall_3 += recall_3

        # AP@3
        ap_3 = 0.0
        hits = 0
        for k, rec_name in enumerate(rec_set[:3], 1):  # Limit to 3 for AP@3
            if rec_name in relevant_set:
                hits += 1
                ap_3 += hits / k
        ap_3 = ap_3 / min(3, relevant_count) if relevant_count > 0 else 0.0
        mean_ap_3 += ap_3

    mean_recall_3 /= n
    mean_ap_3 /= n

    return mean_recall_3, mean_ap_3

if __name__ == "__main__":
    from data_preprocessing import load_and_preprocess_data
    df, model, index = load_and_preprocess_data()
    query = "We are seeking a Junior Java Developer to join our development team. The ideal candidate will be responsible for writing clean, efficient Java code, collaborating with business analysts and product managers to design software solutions, and participating in code reviews. Key tasks include developing and maintaining web applications using Java frameworks (e.g., Spring), troubleshooting and debugging applications, and working effectively in a team environment to meet project deadlines."
    recs = recommend_assessments(query, df, model, index, max_duration=40, preferred_test_types=['K', 'C'])
    print("Recommendations:", recs)

    queries = ["Hiring for Java developers with collaboration skills", "Administrative professional test"]
    relevant = [["Java Web Services (New)", "Java 8 (New)"], ["Administrative Professional - Short Form"]]
    recall, ap = evaluate_recommendations(queries, relevant, df, model, index)
    print(f"Mean Recall@3: {recall:.4f}, Mean AP@3: {ap:.4f}")