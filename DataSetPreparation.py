import pandas as pd
from bs4 import BeautifulSoup
import requests

# Load the existing CSV file
input_file = "data/SHL_merged.csv"
df = pd.read_csv(input_file)

# Function to scrape data from URL
def scrape_assessment_details(url):
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        # # Extract Job Description
        desc_div = soup.find('div', class_='product-catalogue-training-calendar__row typ')
        job_desc = desc_div.find('h4', string="Description").find_next('p').get_text(strip=True) if desc_div and desc_div.find('h4', string="Description") else ""

        # Extract Duration by checking all matching divs
        print("Started")
        duration = None
        all_rows = soup.find_all('div', class_='product-catalogue-training-calendar__row typ')
        for row in all_rows:
            h4_tag = row.find('h4', string="Assessment length")
            if h4_tag:
                p_tag = h4_tag.find_next('p')
                if p_tag:
                    duration_text = p_tag.get_text(strip=True)
                    print(duration_text)
                    # Keep the full duration text (e.g., "15 to 35")
                    if '=' in duration_text:
                        duration = duration_text.split('=')[-1].strip()
                        print(duration)
                    break  # Stop after finding the first match

        return {
            'Job Description': job_desc,
            'Duration': duration
        }

    except requests.RequestException as e:
        print(f"Network error scraping {url}: {e}")
        return {
            'Job Description': None,
            'Duration': None
            }
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return {
            'Job Description': None,
            'Duration': None
            }

# Apply scraping to each row and create new columns
# df[['Job Description', 'Job Level', 'Duration']] = df['URL'].apply(scrape_assessment_details).apply(pd.Series)
df[['Job Description', 'Duration']] = df['URL'].apply(scrape_assessment_details).apply(pd.Series)

# Save to a new CSV file
output_file = "data/SHL_Final_enriched_Data.csv"
df.to_csv(output_file, index=False)
print(f"New CSV file saved as {output_file}")
