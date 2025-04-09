# SHL Assessment Recommendation System

This repository contains a web application designed to recommend SHL assessments based on natural language queries or job descriptions. It leverages semantic embeddings, keyword scoring, and filtering to provide relevant assessments from SHL’s Product Catalog, addressing inefficiencies in traditional keyword-based searches.

## Project Overview

The system takes a job description (text or URL) and returns up to 10 SHL assessments with attributes: Assessment Name, URL, Remote Testing Support, Adaptive/IRT Support, Duration, and Test Type. It features a Streamlit frontend, FastAPI backend, and is evaluated using Mean Recall@3 and MAP@3 metrics.

### Features
- **Input**: Natural language query or job description URL.
- **Output**: Tabular list of 1–10 assessments with clickable URLs.
- **Evaluation**: Recall@5 and MAP@5 computed in the demo.
- **Deployment**: Hosted on streamlit (free tier).

## Repository Structure
- `webScrapingPrepackeged.py`: Scrapes pre-packaged solutions (12 pages).
- `webScrapingIndividualPackage.py`: Scrapes individual test solutions (32 pages).
- `mergeboth.py`: Merges scraped CSVs.
- `datasetPreparation.py`: Enriches data with Job Description and Duration.
- `dataPreprocessing.py`: Preprocesses data and builds FAISS index.
- `recommendation_engine.py`: Core recommendation logic with evaluation.
- `app.py`: Streamlit frontend for demo.
- `api.py`: FastAPI backend for JSON recommendations.
- `test_api.py`: Test script for API endpoint.
- `data/`: Directory for CSVs (e.g., `SHL_Final_enriched_Data.csv`).

## Setup Instructions

### Prerequisites
- Python 3.8+
- Chrome browser (for Selenium scraping)
- Git

### Installation
1. Clone the repository from GitHub.
2. Install dependencies from `requirements.txt`.
3. Scrape data:
   - Run `webScrapingPrepackeged.py` and `webScrapingIndividualPackage.py` to generate initial CSVs.
   - Merge with `mergeboth.py`.
   - Enrich with `datasetPreparation.py` to create `data/SHL_Final_enriched_Data.csv`.
4. Preprocess data with `dataPreprocessing.py` to verify loading and indexing.
5. Check the `Recommendation_engine.py` Verify running it by removing the comments of '__init__' at last

### Running Locally
- **Streamlit Demo**: Launch `app.py` and open `http://localhost:8501`.
- **FastAPI Server**: Start `api.py` on port 7000 and test with `test_api.py` or a tool like Postman.

## Usage
1. **Demo**: Enter a query (e.g., "Hiring Java developers, 40 mins") or URL in the Streamlit UI, click "Recommend," and view results. Use "Run Evaluation" for accuracy metrics.
2. **API**: Send a POST request to `/recommend`:
   ```json
   {
     "query_text": "Java developers with collaboration skills",
     "max_duration": 40,
     "preferred_test_types": ["K", "C"],
     "top_n": 5
   }
## Approach
- **Data**: Scraped SHL Catalog with Selenium, enriched with BeautifulSoup.  
- **Preprocessing**: Used SentenceTransformer ("all-MiniLM-L6-v2") for embeddings, indexed with FAISS.  
- **Recommendation**: Combined cosine similarity (70%) and skill scoring (30%), filtered by duration/test types.  
- **Deployment**: Streamlit (frontend), FastAPI (API), hosted on Streamlit.  

## Deployment Links
- **Demo**: [https://rohithmacharla11-assessment-recommendation-system-app-sbdauy.streamlit.app/]  
- **API**: [https://github.com/RohithMacharla11/SHL-Assessment-Recommendation-System]  
- **GitHub**: [https://github.com/RohithMacharla11/SHL-Assessment-Recommendation-System]  

## Evaluation
- **Metrics**: Mean Recall@3 and MAP@3 (code in `recommendation_engine.py`).  
- **Demo**: Computes Recall@5 and MAP@5 dynamically.  

## Contributing
Feel free to fork, submit issues, or PRs. Ensure tests pass and dependencies are updated.

## Author
Rohith Macharla
