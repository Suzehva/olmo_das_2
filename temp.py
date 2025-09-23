import json
import re
import collections
import os

USER = 'aditijb'
EXPERIMENT_NAME = "year_localization"
model_name = 'OLMo-2-0425-1B'
revision = 'main'

raw_example_file = f"/nlp/scr/{USER}/olmo_das_2/{EXPERIMENT_NAME}/split_to_raw_example_year_{model_name}-revision{revision}.json"

# Load JSON
with open(raw_example_file, "r") as f:
    split_to_raw_example = json.load(f)

# Tense mapping
tense_map = {"was": "past", "were": "past", "is": "present", "are": "present", "will": "future"}

year_re = re.compile(r'In (\d{4}),')
def get_last_digit(text):
    match = year_re.search(text)
    return int(match.group(1)) % 10 if match else None

# -----------------------------
# Conditional tables
# -----------------------------
def build_conditional_table(data, collapse_tense=True):
    table = collections.defaultdict(lambda: collections.Counter())
    for split, examples in data.items():
        if not isinstance(examples, list):
            continue
        for ex in examples:
            digit = get_last_digit(ex["input"])
            if digit is not None:
                label = ex["inv_label"].strip()
                if collapse_tense and label in tense_map:
                    table[digit][tense_map[label]] += 1
                elif not collapse_tense:
                    table[digit][label] += 1
    return table

def format_table(table, collapse_tense=True):
    if collapse_tense:
        header = ["Digit", "P(past)", "P(present)", "P(future)"]
        tenses = ["past", "present", "future"]
    else:
        all_verbs = sorted({v for counts in table.values() for v in counts.keys()})
        header = ["Digit"] + all_verbs
        tenses = all_verbs

    out_lines = []
    for digit in sorted(table.keys()):
        total = sum(table[digit].values())
        row = [str(digit)]
        for t in tenses:
            count = table[digit][t]
            row.append(f"{count} ({count/total:.3f})" if total > 0 else "0 (0.000)")
        out_lines.append(row)
    return header, out_lines

def print_table(header, out_lines):
    col_widths = [max(len(row[i]) for row in out_lines + [header]) for i in range(len(header))]
    def format_row(row):
        return "  ".join(val.ljust(col_widths[i]) for i, val in enumerate(row))
    print(format_row(header))
    for r in out_lines:
        print(format_row(r))

# -----------------------------
# Train/Test conditional table
# -----------------------------
def counts_past_pres_future_train_test(data):
    table_train = collections.defaultdict(lambda: collections.Counter())
    table_test = collections.defaultdict(lambda: collections.Counter())

    for split, examples in data.items():
        if not isinstance(examples, list):
            continue
        for ex in examples:
            digit = get_last_digit(ex["input"])
            if digit is not None:
                label = ex["inv_label"].strip()
                if label in tense_map:
                    target_table = table_train if split == "das-train" else table_test
                    target_table[digit][tense_map[label]] += 1

    def build_table(table):
        header = ["Digit", "P(past)", "P(present)", "P(future)"]
        out_lines = []
        for digit in sorted(table.keys()):
            total = sum(table[digit].values())
            row = [str(digit)]
            for tense in ["past", "present", "future"]:
                count = table[digit][tense]
                prob = count / total if total > 0 else 0
                row.append(f"{prob:.3f} ({count})")
            out_lines.append(row)
        return header, out_lines

    header_train, out_lines_train = build_table(table_train)
    header_test, out_lines_test = build_table(table_test)

    # print to console
    print(f"\n=== TRAIN vs TEST conditional probs ===")
    print("TRAIN SET (das-train):")
    if out_lines_train:
        print_table(header_train, out_lines_train)
    else:
        print("No training examples found.")
    print("\nTEST SET (all other splits):")
    if out_lines_test:
        print_table(header_test, out_lines_test)
    else:
        print("No test examples found.")

    # write to file
    out_path = f'/nlp/scr/{USER}/olmo_das_2/{EXPERIMENT_NAME}/metadata/last_year_token_conditional_probs_train_vs_test.txt'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(f"Model: {model_name}, Revision: {revision}\n\n")
        f.write("TRAIN SET (das-train):\n")
        if out_lines_train:
            f.write("  ".join(header_train) + "\n")
            for row in out_lines_train:
                f.write("  ".join(row) + "\n")
        else:
            f.write("No training examples found.\n")
        f.write("\nTEST SET (all other splits):\n")
        if out_lines_test:
            f.write("  ".join(header_test) + "\n")
            for row in out_lines_test:
                f.write("  ".join(row) + "\n")
        else:
            f.write("No test examples found.\n")
    print(f"\nTrain vs Test results written to {out_path}")

# -----------------------------
# Run all
# -----------------------------
# Full collapse & full verb tables
table_collapse = build_conditional_table(split_to_raw_example, collapse_tense=True)
table_full = build_conditional_table(split_to_raw_example, collapse_tense=False)

print("\n=== Last Year Digit → Past/Present/Future Probabilities ===")
header, out_lines = format_table(table_collapse, collapse_tense=True)
print_table(header, out_lines)

print("\n=== Last Year Digit → All Verb Counts ===")
header2, out_lines2 = format_table(table_full, collapse_tense=False)
print_table(header2, out_lines2)

# Write stats to file
os.makedirs(f"/nlp/scr/{USER}/olmo_das_2/{EXPERIMENT_NAME}/metadata", exist_ok=True)
out_file = f"/nlp/scr/{USER}/olmo_das_2/{EXPERIMENT_NAME}/last_year_token_conditional_probs.txt"
with open(out_file, "w") as f:
    f.write(f"Model: {model_name}, Revision: {revision}\n\n")
    f.write("=== Past/Present/Future ===\n")
    f.write("  ".join(header) + "\n")
    for row in out_lines:
        f.write("  ".join(row) + "\n")
    f.write("\n=== All Verbs ===\n")
    f.write("  ".join(header2) + "\n")
    for row in out_lines2:
        f.write("  ".join(row) + "\n")
print(f"\nResults written to {out_file}")

# Train vs Test
counts_past_pres_future_train_test(split_to_raw_example)
