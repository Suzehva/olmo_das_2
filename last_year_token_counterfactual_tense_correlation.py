## This file tests the correlation between the last year token and the counterfactual tense label

import json
import re
import os
import collections
from scipy.stats import chi2_contingency

USER = 'aditijb'
model_name = 'OLMo-2-0425-1B'


# EXPERIMENT_NAME = 'year_localization_AI2_full_model' 
# revision = 'main'

EXPERIMENT_NAME = 'year_localization_AI2_10k_model'
revision = 'stage1-step10000-tokens21B'

FILEPATH = f'/nlp/scr/{USER}/olmo_das_2/{EXPERIMENT_NAME}/metadata/split_to_raw_example_year_{model_name}-revision{revision}.json'

# regex to extract year
year_re = re.compile(r'In (\d{4}),')

def get_last_digit(text):
    match = year_re.search(text)
    if match:
        return int(match.group(1)) % 10  # last digit
    return None

with open(FILEPATH, "r") as f:
    data = json.load(f)

# collect last-digit → tense counts
counts = collections.Counter()
table = collections.defaultdict(lambda: collections.Counter())

for split, examples in data.items():
    if not isinstance(examples, list):
        continue  # skip non-list entries
    for ex in examples:
        year_last_digit = get_last_digit(ex["input"])
        if year_last_digit is not None:
            inv_label = ex["inv_label"].strip()
            counts[(year_last_digit, inv_label)] += 1
            table[year_last_digit][inv_label] += 1

# convert to contingency table
all_labels = sorted({lbl for _, lbl in counts.keys()})
digits = sorted(table.keys())
matrix = [[table[d][lbl] for lbl in all_labels] for d in digits]

chi2, p, dof, expected = chi2_contingency(matrix)

# WRITE TO FILE
out_path = f'/nlp/scr/{USER}/olmo_das_2/{EXPERIMENT_NAME}/metadata/last_year_token_counterfactual_tense_correlation.txt'
with open(out_path, "w") as f:
    f.write(f"Digits: {digits}\n")
    f.write(f"Labels: {all_labels}\n")
    f.write("Contingency matrix:\n")
    for row in matrix:
        f.write(f"{row}\n")
    f.write(f"\nChi2={chi2:.3f}, p={p:.5f}, dof={dof}\n")

print(f"Results written to {out_path}")

# PRINT
print("Digits:", digits)
print("Labels:", all_labels)
print("Contingency matrix:")
for row in matrix:
    print(row)
print(f"\nChi2={chi2:.3f}, p={p:.5f}, dof={dof}")


# The json files look like this: and the last year token will be "4" for 2034 and "1" for 2041, etc. the counterfactual tense label is the "inv label" provided. please write a simple correlation script. here is the filepath

# {
#   "das-train": [
#     {
#       "input": "In 2034, there",
#       "label": " were",
#       "source_input": "In 2044, there",
#       "source_label": " is",
#       "inv_label": " are",
#       "split": "BASE_TEMPLATE",
#       "source_split": "BASE_TEMPLATE"
#     },
#     {
#       "input": "In 2041, there",
#       "label": " are",
#       "source_input": "In 1980, there",
#       "source_label": " were",
#       "inv_label": " were",
#       "split": "BASE_TEMPLATE",
#       "source_split": "BASE_TEMPLATE"
#     },
#     {
#       "input": "In 1963, there",
# ...
#   ],
#   "source-In 2014, there-correct-test": [
#     {
#       "input": "In 2030, there",
#       "label": " will",
#       "source_input": "In 2014, there",
#       "source_label": " were",
#       "inv_label": " was",
#       "split": "BASE_TEMPLATE",
#       "source_split": "BASE_TEMPLATE"
#     },
#     {
#       "input": "In 2043, there",
#       "label": " is",
#       "source_input": "In 2014, there",
#       "source_label": " were",
#       "inv_label": " was",
#       "split": "BASE_TEMPLATE",
#       "source_split": "BASE_TEMPLATE"
#     },
#     {
#       "input": "In 2049, there",