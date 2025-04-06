import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

# Setup headless browser
options = Options()
options.add_argument("--headless")
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=options)

data = []

# Go through 32 pages with ?start=0,12,24,...,372
for start in range(0, 384, 12):
    url = f"https://www.shl.com/solutions/products/product-catalog/?start={start}&type=1"
    driver.get(url)
    time.sleep(2)

    # Locate the only table on this type=1 page (Individual Test Solutions)
    rows = driver.find_elements(By.XPATH, "//table//tr")

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

        # Remote Testing
        try:
            span1 = cols[1].find_element(By.TAG_NAME, "span")
            remote_testing = "Yes" if "-yes" in span1.get_attribute("class") else "No"
        except:
            remote_testing = "No"

        # Adaptive/IRT Support
        try:
            span2 = cols[2].find_element(By.TAG_NAME, "span")
            adaptive_support = "Yes" if "-yes" in span2.get_attribute("class") else "No"
        except:
            adaptive_support = "No"

        # Test Type
        test_type_elements = cols[3].find_elements(By.TAG_NAME, "span")
        test_types = "".join([elem.text.strip() for elem in test_type_elements])

        data.append([name, link, remote_testing, adaptive_support, test_types])

# Save to CSV
df = pd.DataFrame(data, columns=["Assessment Name", "URL", "Remote Testing", "Adaptive/IRT Support", "Test Type"])
df.to_csv("data/shl_individual_test_solutions.csv", index=False)

print("✅ Saved Individual Test Solutions to 'shl_individual_test_solutions.csv'")

driver.quit()
