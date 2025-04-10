from sentence_transformers import SentenceTransformer
import pandas as pd
import faiss
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from typing import List, Dict
from recommendation_engine import recommend_assessments  # Import from recommend_engine.py

# Initialize FastAPI app
app = FastAPI(title="SHL Assessment Recommender API")

# Global variables for data, model, and index
df, model, index = None, None, None

@app.on_event("startup")
def startup_event():
    global df, model, index
    from data_preprocessing import load_and_preprocess_data  # Assuming this exists
    try:
        df, model, index = load_and_preprocess_data()
        print("Data, model, and index loaded successfully!")
    except Exception as e:
        print(f"Error during startup: {str(e)}")
        raise  # Fail startup if loading fails

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Define input schema for /recommend
class QueryRequest(BaseModel):
    query: str
    max_duration: float | None = None  # Optional field
    preferred_test_types: List[str] | None = None  # Optional field

# Updated /recommend endpoint using recommend_assessments from recommend_engine
@app.post("/recommend")
async def get_recommendations(request: QueryRequest) -> Dict[str, List[Dict]]:
    try:
        # Call recommend_assessments from recommend_engine.py
        recommendations = recommend_assessments(
            query_text=request.query,
            df=df,
            model=model,
            index=index,
            max_duration=request.max_duration,
            preferred_test_types=request.preferred_test_types,
            top_n=10  # Max 10 as per your API spec
        )

        # If no recommendations, return a default assessment (min 1 requirement)
        if not recommendations:
            default_assessment = {
                "url": "https://www.shl.com/solutions/products/product-catalog/",
                "adaptive_support": "No",
                "description": "No matching assessments found",
                "duration": 0,
                "remote_support": "No",
                "test_type": ["Unknown"]
            }
            recommendations = [default_assessment]

        # Transform results to match API spec
        formatted_recommendations = []
        for rec in recommendations:
            # Parse duration to integer, handle 'N/A' or invalid cases
            try:
                duration = int(float(rec['Duration'])) if rec['Duration'] != 'N/A' else 0
            except (ValueError, TypeError):
                duration = 0

            # Parse test_type to list
            test_type = str(rec['Test Type']).split(',') if rec['Test Type'] else ["Unknown"]
            test_type = [t.strip() for t in test_type]

            formatted_recommendations.append({
                "url": str(rec['URL']),
                "adaptive_support": rec['Adaptive/IRT Support'],
                "description": str(rec['Assessment Name']),
                "duration": duration,
                "remote_support": rec['Remote Testing Support'],
                "test_type": test_type
            })

        return {"recommended_assessments": formatted_recommendations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")
# # This is for Testing locally eg: using POSTMAN 
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))  # Use environment variable PORT or default to 8080
    uvicorn.run(app, host="0.0.0.0", port=port)

# handler = Mangum(app)  # Vercel entry point