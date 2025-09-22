## This file tests the correlation between the last year token and the counterfactual tense label

import json
import re
import os
import collections
from scipy.stats import chi2_contingency


def counts_past_pres_future():
    # regex to extract year
    year_re = re.compile(r'In (\d{4}),')

    def get_last_digit(text):
        match = year_re.search(text)
        if match:
            return int(match.group(1)) % 10
        return None

    # tense grouping map
    tense_map = {
        "was": "past",
        "were": "past",
        "is": "present",
        "are": "present",
        "will": "future",
    }

    with open(FILEPATH, "r") as f:
        data = json.load(f)

    # collect last-digit → tense counts (collapsed to 3 categories)
    table = collections.defaultdict(lambda: collections.Counter())

    for split, examples in data.items():
        if not isinstance(examples, list):
            continue
        for ex in examples:
            year_last_digit = get_last_digit(ex["input"])
            if year_last_digit is not None:
                inv_label = ex["inv_label"].strip()
                if inv_label in tense_map:
                    table[year_last_digit][tense_map[inv_label]] += 1

    # build output lines
    header = ["Digit", "P(past)", "P(present)", "P(future)"]
    out_lines = []

    for digit in sorted(table.keys()):
        total = sum(table[digit].values())
        probs = []
        for tense in ["past", "present", "future"]:
            prob = table[digit][tense] / total if total > 0 else 0
            probs.append(f"{prob:.3f}")
        out_lines.append([str(digit)] + probs)

    # formatting widths
    col_widths = [max(len(row[i]) for row in out_lines + [header]) for i in range(len(header))]

    def format_row(row):
        return "  ".join(val.ljust(col_widths[i]) for i, val in enumerate(row))

    # write to file
    out_path = f'/nlp/scr/{USER}/olmo_das_2/{EXPERIMENT_NAME}/metadata/last_year_token_conditional_probs_per_tense.txt'
    with open(out_path, "w") as f:
        f.write(f"Model: {model_name}, Revision: {revision}\n\n")
        f.write(format_row(header) + "\n")
        f.write("\n".join(format_row(row) for row in out_lines))

    # print to console
    print(f"Model: {model_name}, Revision: {revision}\n")
    print(format_row(header))
    for row in out_lines:
        print(format_row(row))
    print(f"\nResults written to {out_path}")




def counts_was_were_is_are_will():
    # regex to extract year
    year_re = re.compile(r'In (\d{4}),')

    def get_last_digit(text):
        match = year_re.search(text)
        if match:
            return int(match.group(1)) % 10  # last digit of year
        return None

    with open(FILEPATH, "r") as f:
        data = json.load(f)

    # collect last-digit → tense counts
    table = collections.defaultdict(lambda: collections.Counter())

    for split, examples in data.items():
        if not isinstance(examples, list):
            continue
        for ex in examples:
            year_last_digit = get_last_digit(ex["input"])
            if year_last_digit is not None:
                inv_label = ex["inv_label"].strip()
                table[year_last_digit][inv_label] += 1

    # build output lines with aligned spacing
    header = ["Digit", "Tense", "Count", "P(Tense|Digit)"]
    out_lines = []

    for digit in sorted(table.keys()):
        total = sum(table[digit].values())
        for tense, count in table[digit].items():
            prob = count / total if total > 0 else 0
            out_lines.append([str(digit), tense, str(count), f"{prob:.3f}"])

    # formatting widths
    col_widths = [max(len(row[i]) for row in out_lines + [header]) for i in range(len(header))]

    def format_row(row):
        return "  ".join(val.ljust(col_widths[i]) for i, val in enumerate(row))

    # write to file
    out_path = f'/nlp/scr/{USER}/olmo_das_2/{EXPERIMENT_NAME}/metadata/last_year_token_conditional_probs_per_verb.txt'
    with open(out_path, "w") as f:
        f.write(f"Model: {model_name}, Revision: {revision}\n\n")
        f.write(format_row(header) + "\n")
        f.write("\n".join(format_row(row) for row in out_lines))

    # print to console
    print(f"Model: {model_name}, Revision: {revision}\n")
    print(format_row(header))
    for row in out_lines:
        print(format_row(row))
    print(f"\nResults written to {out_path}")



####################################
######## MAIN CODE #################
####################################

USER = 'aditijb'
model_name = 'OLMo-2-0425-1B'


# EXPERIMENT_NAME = 'year_localization_AI2_full_model' 
# revision = 'main'

EXPERIMENT_NAME = 'year_localization_AI2_10k_model'
revision = 'stage1-step10000-tokens21B'

FILEPATH = f'/nlp/scr/{USER}/olmo_das_2/{EXPERIMENT_NAME}/metadata/split_to_raw_example_year_{model_name}-revision{revision}.json'



counts_past_pres_future()
counts_was_were_is_are_will()






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