import collections
import random

import datasets
from datasets import Dataset

_BASE_TEMPLATE = "BASE_TEMPLATE"
_SOURCE_TEMPLATE = "SOURCE_TEMPLATE"
FEATURE_TYPES = datasets.Features(
  {
    "input": datasets.Value("string"),
    "label": datasets.Value("string"),
    "source_input": datasets.Value("string"),
    "source_label": datasets.Value("string"),
    "inv_label": datasets.Value("string"),
    "split": datasets.Value("string"),
    "source_split": datasets.Value("string"),
  }
)


def load_intervention_data(
  mode,
  verified_examples,
  data_split,
  prompt_to_vars,
  inv_label_fn,
  filter_fn=None,
  bos_pad=None,
  max_example_per_split=20480,
  max_example_per_eval_split=10,
):
  
  # REMOVE THIS!!
  # max_example_per_eval_split = 100 # hardcoded for now
  # random.seed(0) # Aditi

  # inv_label_fn: A callable that takes in the variables parsed from the
  # base and source input, i.e., two dictionaries and returns a boolean.
  # verified_examples: data['train']['correct']
  if mode == "verified_example_selection":
    raise ValueError("Verified example selection is deprecated.")
    base_examples = data_split["train"]["correct"]
    source_examples = (
      data_split["train"]["correct"] + data_split["train"]["wrong"]
    )
  elif mode == "localization":
    base_examples = verified_examples
    source_examples = verified_examples
  elif mode == "val_all":
    base_examples = verified_examples
    source_examples = data_split["val"]["correct"] + data_split["val"]["wrong"]
  elif mode == "val_debug":
    base_examples = verified_examples
    source_examples = (
      data_split["val"]["correct"][:10] + data_split["val"]["wrong"][:10]
    )
  elif mode == "test_all":
    base_examples = verified_examples
    source_examples = (
      data_split["test"]["correct"] + data_split["test"]["wrong"]
    )
  elif mode == "das":
    # base_examples are ...['train']['correct']
    # source examples are from ...['train']['correct'] + ...['test']
    base_examples = verified_examples
    source_examples = (
      base_examples + data_split["val"]["correct"] + data_split["val"]["wrong"]
    )

  source_example_calculation = ""
  if mode == "das":
    source_example_calculation = (
      f"{len(base_examples)}+{len(data_split['test']['correct'])}+"
      f"{len(data_split['test']['wrong'])}={len(source_examples)}"
    )
  elif mode == "test_all" or mode == "val_all":
    s = mode.split("_all")[0]
    source_example_calculation = (
      f"{len(data_split[s]['correct'])}+{len(data_split[s]['wrong'])}="
      f"{len(source_examples)}"
    )
  print(
    f"mode={mode}, "
    f"#base_examples={len(base_examples)}, "
    f"#source_examples={source_example_calculation}"
  )

  train_num_calculation, val_num_calculation, test_num_calculation = "", "", ""
  # gathers all pairs of (base, source) examples into a dictionary
  # keyed by their “split key.”
  split_to_raw_example = collections.defaultdict(list)
  for j in range(len(source_examples)):
    for i in range(len(base_examples)):
      base_vars = prompt_to_vars[base_examples[i]]
      source_vars = prompt_to_vars[source_examples[j]]
      # print(f"DEBUG: base_vars={base_vars}, source_vars={source_vars}")

      if filter_fn and not filter_fn(base_vars, source_vars):
        # print("DEBUG: skipped because of the filter function")
        # print(f" >>  base: {base_vars}, src: {source_vars}")

        # TODO: Aditi (Sept 18 2025)
        # instead of only not including it, we want to skip this base/source pairing and just match the next available base/source for a total of 10
        # for some reason the base/source pairings are not adding up to a total of 10 for certain years.
        continue

      # Set split.
      # split_key = "...-train" or "...-val" or "...-test"
      # each key is a “split identifier,” and
      # the values are lists of examples (dictionaries)
      # {
      # "das-train": [ { ... }, { ... }, ... ],
      # "source-foo-correct-test": [ { ... }, ... ],
      # ...
      # }
      src_is_correct = any(
        source_examples[j] in data_split[s]["correct"] for s in data_split
      )
      split_key = (
        f"source-{source_examples[j]}-"
        f"{'correct' if src_is_correct else 'wrong'}-test"
      )
      # Before-split formulas
      test_num_calculation = (
        f"{len(base_examples)}*({len(data_split['test']['correct'])}+"
        f"{len(data_split['test']['wrong'])})"
      )
      if mode == "verified_example_selection":
        split_key = f"base-{base_examples[i]}-{'correct' if source_examples[j] in data_split['train']['correct'] else 'wrong'}-test"
      if mode == "das":
        # base_examples[i] is always in ...['train']['correct']
        # source_examples[j] can be in ...['train']['correct'] or
        #   ...['test']['correct'] or ...['test']['wrong']
        if (
          base_examples[i] in data_split["train"]["correct"]
          and source_examples[j] in data_split["train"]["correct"]
        ):
          split_key = "das-train"
        elif (
          base_examples[i] in data_split["train"]["correct"]
          and source_examples[j] in data_split["val"]["correct"]
        ):
          split_key = f"source-{source_examples[j]}-correct-test"
        elif (
          base_examples[i] in data_split["train"]["correct"]
          and source_examples[j] in data_split["val"]["wrong"]
        ):
          split_key = f"source-{source_examples[j]}-wrong-test"
        else:
          continue
        train_num_calculation = f"{len(base_examples)}*{len(base_examples)}"
        val_num_calculation = ""
        test_num_calculation = (
          f"{len(base_examples)}*({len(data_split['test']['correct'])}+"
          f"{len(data_split['test']['wrong'])})"
        )
      if i < 3 and j < 3:
        print(f"base_vars={base_vars}, source_vars={source_vars}")
      split_to_raw_example[split_key].append(
        {
          "input": base_examples[i],
          "label": base_vars["label"],
          "source_input": source_examples[j],
          "source_label": source_vars["label"],
          "inv_label": inv_label_fn(base_vars, source_vars),
          # Determine the intervention locations.
          "split": base_vars["split"], # says "_BASE_TEMPLATE"
          "source_split": source_vars["split"],
        }
      )
      if mode == "val_debug" and i == 0:
        # Add a no op where base = source.
        split_to_raw_example[split_key] = [
          {
            "input": source_examples[j],
            "label": source_vars["label"],
            "source_input": source_examples[j],
            "source_label": source_vars["label"],
            "inv_label": inv_label_fn(source_vars, source_vars),
            # Determine the intervention locations.
            "split": source_vars["split"],
            "source_split": source_vars["split"],
          }
        ]
  split_to_raw_example = dict(split_to_raw_example)
  bos_pad = bos_pad or ""

  for split in split_to_raw_example:
    for i in range(len(split_to_raw_example[split])):
      split_to_raw_example[split][i]["inv_label"] = (
        bos_pad + split_to_raw_example[split][i]["inv_label"]
      )
      split_to_raw_example[split][i]["label"] = (
        bos_pad + split_to_raw_example[split][i]["label"]
      )

  # Preprocess the dataset.
  for split in split_to_raw_example:
    # Shuffle examples
    random.shuffle(split_to_raw_example[split])
  # Remove empty splits.
  split_to_raw_example = {
    k: v for k, v in split_to_raw_example.items() if len(v) > 0
  }
  # These counts reflect the raw examples found in split_to_raw_example—before
  # any further subsampling.
  print(
    f"BEFORE SPLIT: "
    f"#Training examples={train_num_calculation}={sum(map(len, [v for k, v in split_to_raw_example.items() if k.endswith('-train')]))}, "
    f"#Validation examples={val_num_calculation}={sum(map(len, [v for k, v in split_to_raw_example.items() if k.endswith('-val')]))}, "
    f"#Test examples={test_num_calculation}={sum(map(len, [v for k, v in split_to_raw_example.items() if k.endswith('-test')]))}"
  )

  split_to_dataset = {
    split: Dataset.from_list(
      [
        x
        for x in split_to_raw_example[split][
          : max_example_per_eval_split  # 10
          if mode == "localization"
          or (mode == "das" and split.endswith("-test"))
          else max_example_per_split # for training
        ]
        # TODO: Aditi
        # should we remove the limit of 10 (max_example_per_eval_split) so we use as many as possible for test, and
        # similarly for max_example_per_split so we use many for train?
      ],
      features=FEATURE_TYPES,
    )
    for split in split_to_raw_example
  }

  if mode == "das":
    train_num_calculation_after = (
      f"min({len(base_examples)}*{len(base_examples)}, 20480)"
    )
    test_num_calculation_after = (
      f"{len(data_split['test']['correct'])}*10+"
      f"{len(data_split['test']['wrong'])}*10"
    )
    val_num_calculation_after = ""
  else:
    train_num_calculation_after = ""
    test_num_calculation_after = (
      f"({len(data_split['test']['correct'])}+"
      f"{len(data_split['test']['wrong'])})*10"
    )
    val_num_calculation_after = ""

  print(
    f"AFTER SPLIT KEPT: "
    f"#Training examples={train_num_calculation_after}="
    f"{sum(map(len, [v for k, v in split_to_dataset.items() if k.endswith('-train')]))}, "
    f"#Validation examples={val_num_calculation_after}="
    f"{sum(map(len, [v for k, v in split_to_dataset.items() if k.endswith('-val')]))}, "
    f"#Test examples={test_num_calculation_after}="
    f"{sum(map(len, [v for k, v in split_to_dataset.items() if k.endswith('-test')]))}"
  )
  print(
    f"#Splits=1 training split plus 1,024 "
    f"test splits={len(split_to_raw_example)}"
  )
  return split_to_raw_example, split_to_dataset