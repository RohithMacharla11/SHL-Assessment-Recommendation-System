# api.py
from sentence_transformers import SentenceTransformer, util
import numpy as np
import pandas as pd
import faiss
from typing import List, Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os

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

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Define input schema for /recommend
class QueryRequest(BaseModel):
    query: str

# Recommendation function (updated to match required response)
def recommend_assessments(query: str, df: pd.DataFrame, model: SentenceTransformer) -> List[Dict]:
    if not query or not isinstance(query, str):
        # Return a default assessment if no valid query (min 1 requirement)
        default_assessment = df.iloc[0] if len(df) > 0 else {
            "url": "https://www.shl.com/solutions/products/product-catalog/",
            "adaptive_support": "No",
            "description": "Default assessment due to invalid query",
            "duration": 0,
            "remote_support": "No",
            "test_type": ["Unknown"]
        }
        return [default_assessment]

    query_keywords = set(query.lower().split())
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
        query_embedding = model.encode([query], convert_to_tensor=False, show_progress_bar=False)
        if len(query_embedding) == 0:
            return [df.iloc[0]]  # Default to first row if encoding fails
    except Exception as e:
        print(f"Error generating query embedding: {str(e)}")
        return [df.iloc[0]]  # Default to first row

    required_columns = ['Job Description', 'Assessment Name', 'Test Type', 'Remote Testing', 'Adaptive/IRT Support', 'Required Skills', 'URL', 'Duration']
    for col in required_columns:
        if col not in df.columns:
            df[col] = "" if col != 'Duration' else 0
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
        duration = row.get('Duration', 0)  # Default to 0 if missing
        try:
            duration = int(float(duration)) if duration != 'N/A' else 0  # Ensure integer
        except (ValueError, TypeError):
            duration = 0

        test_types = str(row.get('Test Type', '')).split(',') if row.get('Test Type') else ["Unknown"]
        test_types = [t.strip() for t in test_types]  # Convert to list

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
            "url": str(row.get('URL', 'https://www.shl.com/solutions/products/product-catalog/')),
            "adaptive_support": "Yes" if row.get('Adaptive/IRT Support', False) else "No",
            "description": str(row.get('Assessment Name', '')),  # Use Assessment Name as description
            "duration": duration,
            "remote_support": "Yes" if row.get('Remote Testing', False) else "No",
            "test_type": test_types,
            "_score": combined_score
        })

    results.sort(key=lambda x: x['_score'], reverse=True)
    top_results = results[:min(10, len(results))]  # Max 10
    return top_results if top_results else [results[0]]  # Min 1

# Updated /recommend endpoint
@app.post("/recommend")
async def get_recommendations(request: QueryRequest):
    try:
        recommendations = recommend_assessments(
            query=request.query,
            df=df,
            model=model
        )
        return {"recommended_assessments": recommendations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 7000))  # For Render compatibility
    uvicorn.run(app, host="0.0.0.0", port=port)