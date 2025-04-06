import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import Tuple

# Function to convert duration strings to numeric values with improved handling
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

# Load and preprocess data with error handling and enhanced embedding
def load_and_preprocess_data(input_file: str = "SHL_Final_enriched_Data.csv") -> Tuple[pd.DataFrame, SentenceTransformer, faiss.IndexFlatL2]:
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

if __name__ == "__main__":
    try:
        df, model, index = load_and_preprocess_data()
        print("Preprocessed data sample:")
        print(df[['Assessment Name', 'Duration', 'Job Description', 'Test Type']].head())
    except Exception as e:
        print(f"Failed to run preprocessing: {str(e)}")