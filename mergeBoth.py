import pandas as pd

# Load the two CSV files
csv1 = pd.read_csv("shl_prepackaged_assessments.csv")
csv2 = pd.read_csv("shl_individual_test_solutions.csv", encoding='ISO-8859-1')

# Append the second file to the first
merged = pd.concat([csv1, csv2], ignore_index=True)

# Save the merged file
merged.to_csv("SHL_merged.csv", index=False)

print("CSV files merged successfully into 'SHL_merged.csv'")
