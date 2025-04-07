# api.py
from sentence_transformers import SentenceTransformer, util
import numpy as np
import pandas as pd
import faiss
from typing import List, Dict, Optional, Tuple
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI(title="SHL Assessment Recommender API")

# Load data, model, and index at startup
df, model, index = None, None, None

@app.on_event("startup")
def startup_event():
    global df, model, index
    from data_preprocessing import load_and_preprocess_data
    df, model, index = load_and_preprocess_data()
    print("Data, model, and index loaded successfully!")

# Define input schema using Pydantic
class QueryRequest(BaseModel):
    query_text: str
    max_duration: Optional[float] = None
    preferred_test_types: Optional[List[str]] = None
    top_n: Optional[int] = 10

# Recommendation function
def recommend_assessments(query_text: str, df: pd.DataFrame, model: SentenceTransformer, 
                         index: Optional[object] = None, max_duration: Optional[float] = None, 
                         preferred_test_types: Optional[List[str]] = None, top_n: int = 10) -> List[Dict]:
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
    skill_hints = {
        'java': 1.0, 'spring': 0.9, 'web': 0.8, 'debugging': 0.7, 'collaboration': 0.6,
        'communication': 0.6, 'teamwork': 0.5, 'framework': 0.7
    }
    inferred_test_types = [test_type_hints.get(kw) for kw in query_keywords if kw in test_type_hints]
    inferred_test_types = list(set(t for t in inferred_test_types if t)) or None
    query_skills = {kw: skill_hints.get(kw, 0.2) for kw in query_keywords if kw in skill_hints or kw in test_type_hints}

    try:
        query_embedding = model.encode([query_text], convert_to_tensor=False, show_progress_bar=False)
        if len(query_embedding) == 0:
            return []
    except Exception as e:
        print(f"Error generating query embedding: {str(e)}")
        return []

    required_columns = ['Job Description', 'Assessment Name', 'Test Type', 'Remote Testing', 'Adaptive/IRT Support', 'Required Skills']
    for col in required_columns:
        if col not in df.columns:
            df[col] = ""
    df['text_for_embedding'] = (
        "Job Description: " + df["Job Description"].fillna("") + " [weight: 0.5]. " +
        "Assessment Name: " + df["Assessment Name"].fillna("") + " [weight: 0.3]. " +
        "Test Type: " + df["Test Type"].fillna("") + " [weight: 0.15]. " +
        "Required Skills: " + df.get("Required Skills", pd.Series([""] * len(df))).fillna("") + " [weight: 0.05]"
    )
    column_embeddings = model.encode(df["text_for_embedding"].tolist(), convert_to_tensor=False, show_progress_bar=False)

    query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1)[:, np.newaxis]
    column_embeddings = column_embeddings / np.linalg.norm(column_embeddings, axis=1)[:, np.newaxis]

    similarities = util.cos_sim(query_embedding, column_embeddings)[0].numpy()

    results = []
    for i in range(len(df)):
        row = df.iloc[i]
        duration = row.get('Duration', float('inf'))

        if max_duration is not None and duration > (max_duration * 1.1):
            continue

        effective_test_types = preferred_test_types or inferred_test_types
        test_types = str(row.get('Test Type', '')).replace(' ', '').lower()
        if effective_test_types and not any(t.lower() in test_types for t in effective_test_types):
            continue

        cosine_score = similarities[i]
        skill_match_score = 0.0
        row_text = (str(row.get('Job Description', '')) + " " + 
                    str(row.get('Assessment Name', '')) + " " + 
                    str(row.get('Required Skills', ''))).lower()
        for skill, weight in query_skills.items():
            if skill in row_text:
                skill_match_score += weight
        combined_score = 0.6 * cosine_score + 0.4 * skill_match_score

        results.append({
            'Assessment Name': row.get('Assessment Name', ''),
            'URL': row.get('URL', ''),
            'Remote Testing Support': 'Yes' if row.get('Remote Testing', False) else 'No',
            'Adaptive/IRT Support': 'Yes' if row.get('Adaptive/IRT Support', False) else 'No',
            'Test Type': row.get('Test Type', ''),
            'Duration': str(duration) if not np.isinf(duration) else 'N/A',
            '_score': combined_score
        })

    results.sort(key=lambda x: x['_score'], reverse=True)
    return [{k: v for k, v in r.items() if k != '_score'} for r in results[:top_n]]

# API endpoint to get recommendations
@app.post("/recommend", response_model=List[Dict])
async def get_recommendations(request: QueryRequest):
    try:
        recommendations = recommend_assessments(
            query_text=request.query_text,
            df=df,
            model=model,
            index=index,
            max_duration=request.max_duration,
            preferred_test_types=request.preferred_test_types,
            top_n=request.top_n
        )
        if not recommendations:
            raise HTTPException(status_code=404, detail="No recommendations found for the given query.")
        return recommendations
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

# Root endpoint for testing
@app.get("/")
async def root():
    return {"message": "Welcome to the SHL Assessment Recommender API!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7000)