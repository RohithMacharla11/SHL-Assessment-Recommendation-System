from flask import Flask, request, jsonify
from data_preprocessing import load_and_preprocess_data
from recommendation_engine import recommend_assessments
import pandas as pd
import re
import logging
from flask_cors import CORS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

try:
    df, model, index = load_and_preprocess_data()
    logger.info("Data, model, and index loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load data, model, or index: {str(e)}")
    exit(1)

def extract_filters(job_description: str) -> tuple:
    max_duration_match = re.search(r'(\d+)\s*(?:minutes|mins)', job_description.lower())
    max_duration = int(max_duration_match.group(1)) if max_duration_match else None
    test_type_hints = {...}  # As defined earlier
    keywords = set(job_description.lower().split())
    preferred_test_types = [test_type_hints[kw] for kw in keywords if kw in test_type_hints]
    return max_duration, list(set(preferred_test_types)) if preferred_test_types else None

@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    if request.method == 'POST':
        data = request.get_json()
        job_description = data.get('job_description', '')
    else:
        job_description = request.args.get('job_description', '')

    if not job_description or not isinstance(job_description, str) or len(job_description) > 1000:
        return jsonify({"error": "Invalid or missing job_description (max 1000 chars)"}), 400

    try:
        max_duration, preferred_test_types = extract_filters(job_description)
        recommendations = recommend_assessments(
            job_description, df, model, index,
            max_duration=max_duration,
            preferred_test_types=preferred_test_types,
            top_n=10
        )
        result = {
            "status": "success",
            "job_description": job_description,
            "recommendations": recommendations
        }
        logger.info(f"Recommendations generated for: {job_description}")
        return jsonify(result)
    except Exception as e:
        logger.error(f"Recommendation error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
    
#use PostMan API
"""Set method to POST.
URL: http://localhost:5000/recommend.
Add header Content-Type: application/json.
Body: { "job_description": "Hiring a Junior Java Developer with Spring skills, max 40 minutes" }.
Click Send."""

#GET URL http://localhost:5000/recommend?job_description=Hiring%20a%20Junior%20Java%20Developer%20with%20Spring%20skills%2C%20max%2040%20minutes
#Using curl:  curl -X POST http://localhost:5000/recommend -H "Content-Type: application/json" -d '{"job_description": "Hiring a Junior Java Developer with Spring skills, max 40 minutes"}'
