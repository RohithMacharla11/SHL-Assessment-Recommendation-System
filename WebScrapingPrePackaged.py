import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

# Function to scrape Pre-packaged Job Solutions
def scrape_prepackaged(url, max_pages=12):
    driver.get(url)
    time.sleep(3)
    section_data = []

    for _ in range(max_pages):
        time.sleep(2)

        rows = driver.find_elements(By.XPATH, "//tr")
        for row in rows:
            cols = row.find_elements(By.TAG_NAME, "td")
            if len(cols) < 4:
                continue

            try:
                name_tag = cols[0].find_element(By.TAG_NAME, "a")
                name = name_tag.text.strip()
                link = name_tag.get_attribute("href")
            except:
                name, link = "N/A", "N/A"

            try:
                span1 = cols[1].find_element(By.TAG_NAME, "span")
                remote_testing = "Yes" if "-yes" in span1.get_attribute("class") else "No"
            except:
                remote_testing = "No"

            try:
                span2 = cols[2].find_element(By.TAG_NAME, "span")
                adaptive_support = "Yes" if "-yes" in span2.get_attribute("class") else "No"
            except:
                adaptive_support = "No"

            test_type_elements = cols[3].find_elements(By.TAG_NAME, "span")
            test_types = "".join([elem.text.strip() for elem in test_type_elements])

            section_data.append([name, link, remote_testing, adaptive_support, test_types])

        # Go to next page
        try:
            next_button = driver.find_element(By.XPATH, "//a[contains(text(), 'Next')]")
            driver.execute_script("arguments[0].click();", next_button)
        except:
            break

    return section_data

# Set up headless Chrome
options = Options()
options.add_argument("--headless")
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")

service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=options)

# Scrape Pre-packaged Job Solutions (12 pages)
prepackaged_url = "https://www.shl.com/solutions/products/product-catalog/"
prepackaged_data = scrape_prepackaged(prepackaged_url, 12)

# Save to CSV
df = pd.DataFrame(prepackaged_data, columns=["Assessment Name", "URL", "Remote Testing", "Adaptive/IRT Support", "Test Type"])
df.to_csv("shl_prepackaged_assessments.csv", index=False)

print("✅ Pre-packaged Job Solutions saved to 'shl_prepackaged_assessments.csv'.")

driver.quit()
