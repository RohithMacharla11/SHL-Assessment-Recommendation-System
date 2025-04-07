# test_api.py
import requests

url = "http://localhost:7000/recommend"
payload = {
    "query_text": "Hiring for Java developers with collaboration skills",
    "max_duration": 40,
    "preferred_test_types": ["K", "C"],
    "top_n": 5
}

response = requests.post(url, json=payload)
if response.status_code == 200:
    print("Recommendations:")
    print(response.json())
else:
    print(f"Error: {response.status_code} - {response.text}")