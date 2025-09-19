import collections
import gc

import datasets
import numpy as np
import pyvene as pv
import torch
from tqdm.auto import tqdm
from utils.dataset_utils import get_dataloader, get_label_offset
from utils.intervention_utils import (
  get_intervention_config,
  remove_invalid_token_id,
)


def load_intervenable_with_vanilla_intervention(
  model, reprs, layers, units, num_unit=1
):
  """Interchange interventions with full representations."""
  inv_config = get_intervention_config(
    type(model), reprs, layers, pv.VanillaIntervention, units, num_unit=num_unit
  )
  print(inv_config)
  intervenable = pv.IntervenableModel(inv_config, model)
  intervenable.set_device(model.device)
  intervenable.disable_model_gradients()
  for k in intervenable.interventions:
    intervenable.interventions[k][0].set_interchange_dim(
      interchange_dim=model.config.hidden_size * num_unit
    )
  intervenable.model.eval()
  return intervenable


def binary_comparison_metrics(
  eval_logits, eval_labels, pad_token_id, last_n_tokens=None, **kwargs
):
  del last_n_tokens
  epsilon = kwargs.get("epsilon", 0)
  metrics = {}
  for key, logits in eval_logits.items():
    total_count, correct = 0, 0
    diff = []
    correct_prob = []
    for eval_pred, eval_label in zip(logits, eval_labels):
      eval_pred = torch.nn.functional.softmax(eval_pred, dim=-1)
      correct_labels = eval_label[:, 0]
      wrong_labels = eval_label[:, 1]
      correct_pred = eval_pred[range(eval_pred.shape[0]), -1, correct_labels]
      wrong_pred = eval_pred[range(eval_pred.shape[0]), -1, wrong_labels]
      correct += (
        ((correct_pred - wrong_pred) > epsilon).type(torch.float32).sum()
      )
      total_count += eval_pred.shape[0]
      diff.append(correct_pred - wrong_pred)
      correct_prob.append(correct_pred)
    acc = (correct / total_count).tolist()
    diff = (torch.cat(diff, dim=0)).tolist()
    correct_prob = torch.cat(correct_prob, dim=0).tolist()
    metrics[key] = {
      "accuracy": round(acc, 4),
      "prob_diff": list(map(lambda x: round(x, 4), diff)),
      "conf": list(map(lambda x: round(x, 4), correct_prob)),
    }
  # For compatablity with other metrics.
  metrics["accuracy"] = metrics["inv_outputs"]["accuracy"]
  return metrics


# TODO: Aditi 
# How os IIA calculated
def compute_metrics(
  keyed_eval_preds,
  eval_labels,
  pad_token_id,
  last_n_tokens=1,
  inference_mode=None,
  **kwargs,
):
  """Computes squence-level and token-level accuracy."""
  # print("\n\n COMPUTING ACCURACY \n\n")
  metrics = {}
  for key, eval_preds in keyed_eval_preds.items():
    total_count, total_token_count = 0, 0
    correct_count, correct_token_count = 0, 0
    class_0_correct_count = 0
    class_0_val = eval_labels[0][0, -1]
    for eval_pred, eval_label in zip(eval_preds, eval_labels):

      # total of 16 batches
      # unsure what the size of each batch is
      # print(f"eval_pred = {eval_pred}\neval_label = {eval_label}")

      if inference_mode == "force_decode":
        eval_pred = eval_pred[:, :-1]

      actual_test_labels = eval_label[:, -last_n_tokens:]

      if actual_test_labels.shape[0] == 0:
        continue
      if len(eval_pred.shape) == 3:
        # eval_preds is in the form of logits.
        pred_test_labels = torch.argmax(eval_pred[:, -last_n_tokens:], dim=-1)
      else:
        # eval_preds is in the form of token ids.
        pred_test_labels = eval_pred[:, -last_n_tokens:]

      padding_tokens = torch.logical_or(
        actual_test_labels == pad_token_id, actual_test_labels < 0
      )
      match_tokens = actual_test_labels == pred_test_labels
      correct_labels = torch.logical_or(match_tokens, padding_tokens)
      total_count += len(correct_labels)
      correct_count += torch.all(correct_labels, axis=-1).float().sum().tolist()
      total_token_count += (~padding_tokens).float().sum().tolist()
      correct_token_count += (
        (~padding_tokens & match_tokens).float().sum().tolist()
      )

      # For binary classification, log the actual prediction by comparing to
      # a single-side label
      class_0_labels = (torch.ones_like(actual_test_labels) * class_0_val).to(
        actual_test_labels.dtype
      )
      class_0_match_tokens = class_0_labels == pred_test_labels
      class_0_correct_labels = torch.logical_or(
        class_0_match_tokens, padding_tokens
      )
      class_0_correct_count += (
        torch.all(class_0_correct_labels, axis=-1).float().sum().tolist()
      )

    accuracy = round(correct_count / total_count, 2)
    token_accuracy = round(correct_token_count / max(1, total_token_count), 2)
    class_0_accuracy = round(class_0_correct_count / total_count, 2)
    metrics[key] = {
      "accuracy": accuracy,
      "token_accuracy": token_accuracy,
      "class_0_accuracy": class_0_accuracy,
    }

  # For compatablity with other metrics.
  metrics["accuracy"] = metrics["inv_outputs"]["accuracy"]
  return metrics

def compute_string_based_metrics(
  keyed_eval_preds,
  eval_labels,
  pad_token_id,
  last_n_tokens=1,
  tokenizer=None,
  extract_prediction_fn=None,
  empty_token="<EMPTY>",
  **kwargs,
):
  """Computes squence-level and token-level accuracy."""
  metrics = {}
  # Add another key 'inv_parsed_outputs' as label_match and pred_match
  for key, eval_preds in keyed_eval_preds.items():
    total_count = 0
    correct_count = 0
    for eval_pred, eval_label in zip(eval_preds, eval_labels):
      actual_test_labels = eval_label[:, -last_n_tokens:]
      if len(eval_pred.shape) == 3:
        # Eval_preds is in the form of logits.
        # (batch_size, seq_length, vocab_size) => logits => take argmax
        pred_test_labels = torch.argmax(eval_pred[:, -last_n_tokens:], dim=-1)
      else:
        # Eval_preds is in the form of token ids.
        # (batch_size, seq_length) => token IDs => just slice
        pred_test_labels = eval_pred[:, -last_n_tokens:]
      # Replaces negative or out-of-range IDs with pad or 0
      label_text = tokenizer.batch_decode(
        remove_invalid_token_id(
          token_ids=actual_test_labels, pad_id=tokenizer.pad_token_id
        ),
        skip_special_tokens=True,
      )
      pred_text = tokenizer.batch_decode(
        remove_invalid_token_id(
          token_ids=pred_test_labels, pad_id=tokenizer.pad_token_id
        ),
        skip_special_tokens=True,
      )
      # For each item in the batch, parse out the final label
      for i in range(len(label_text)):
        label_match = extract_prediction_fn(label_text[i])
        pred_match = extract_prediction_fn(pred_text[i])
        if (
          label_match is not None
          and pred_match is not None
          and label_match != empty_token
          and pred_match != empty_token
          and label_match == pred_match
        ):
          correct_count += 1
        total_count += 1
    accuracy = round(correct_count / total_count, 2)
    token_accuracy = 0  # Do not do per-token accuracy in string-based methods.
    metrics[key] = {"accuracy": accuracy, "token_accuracy": token_accuracy}

  # For compatablity with other metrics.
  metrics["accuracy"] = metrics["inv_outputs"]["accuracy"]
  return metrics


def compute_metrics_case_normalized(
  keyed_eval_preds, eval_labels, pad_token_id, last_n_tokens=1, **kwargs
):
  """Computes squence-level and token-level accuracy."""
  metrics = {}
  for key, eval_preds in keyed_eval_preds.items():
    total_count, total_token_count = 0, 0
    correct_count, correct_token_count = 0, 0
    for eval_pred, eval_label in zip(eval_preds, eval_labels):
      actual_test_labels = eval_label[:, -last_n_tokens:]
      if len(eval_pred.shape) == 3:
        # eval_preds is in the form of logits.
        pred_test_labels = torch.argmax(eval_pred[:, -last_n_tokens:], dim=-1)
      else:
        # eval_preds is in the form of token ids.
        pred_test_labels = eval_pred[:, -last_n_tokens:]
      padding_tokens = torch.logical_or(
        actual_test_labels == pad_token_id, actual_test_labels < 0
      )
      match_tokens = actual_test_labels == pred_test_labels
      correct_labels = torch.logical_or(match_tokens, padding_tokens)
      total_count += len(correct_labels)
      correct_count += torch.all(correct_labels, axis=-1).float().sum().tolist()
      total_token_count += (~padding_tokens).float().sum().tolist()
      correct_token_count += (
        (~padding_tokens & match_tokens).float().sum().tolist()
      )
    accuracy = round(correct_count / total_count, 2)
    token_accuracy = round(correct_token_count / total_token_count, 2)
    metrics[key] = {"accuracy": accuracy, "token_accuracy": token_accuracy}
  # For compatablity with other metrics.
  metrics["accuracy"] = metrics["inv_outputs"]["accuracy"]
  return metrics


def extract_tokens_from_output(
  mode, model_outputs, max_new_tokens, prompt_tokens=None
):
  # Model output could either be logits or token_ids depending on the mode.
  if mode == "generate":
    # Output the greedy decoded sequence.
    if prompt_tokens is not None:
      tokens = model_outputs[
        :, prompt_tokens.shape[1] : prompt_tokens.shape[1] + max_new_tokens
      ]
    else:
      tokens = model_outputs[:, -max_new_tokens:]

  elif mode == "forward":
    # Output the topK token.
    tokens = torch.argsort(model_outputs[:, -1, :], dim=-1, descending=True)[
      :, :max_new_tokens
    ]
  elif mode == "force_decode":
    # Output the force decoded sequence.
    tokens = torch.argsort(
      model_outputs[:, -max_new_tokens - 1 : -1, :], dim=-1, descending=True
    )[:, :, 0]
  else:
    raise ValueError(f"Unknown mode: {mode}")
  return tokens


def eval_with_interventions_batched(
  intervenable,
  split_to_dataset,
  split_to_inv_locations,
  tokenizer,
  compute_metrics_fn,
  max_input_length=None,
  max_new_tokens=1,
  eval_batch_size=16,
  debug_print=False,
  inference_mode=None,
  intervention_location_fn=None,
):
  """Fully batched interchange intervention evaluation."""
  if inference_mode is None:
    # Default to generate.
    inference_mode = "generate"
  assert inference_mode in ("generate", "forward", "force_decode")
  if compute_metrics_fn is None:
    compute_metrics_fn = compute_metrics
  split_to_eval_metrics = {}
  padding_offset = get_label_offset(tokenizer)
  num_inv = len(intervenable.interventions)
  # Merge all splits to allow more efficient batching.
  merged_dataset = datasets.concatenate_datasets(
    [split_to_dataset[split] for split in split_to_dataset]
  )
  split_to_index = {}
  offset = 0
  for split in split_to_dataset:
    split_to_index[split] = [offset, len(split_to_dataset[split]) + offset]
    offset += len(split_to_dataset[split])
  if max_input_length is None:
    # Asssume all inputs have the same max length.
    max_input_length = split_to_inv_locations[merged_dataset[0]["split"]][
      "max_input_length"
    ]
  eval_dataloader = get_dataloader(
    merged_dataset,
    tokenizer=tokenizer,
    batch_size=eval_batch_size,
    prompt_max_length=max_input_length,
    output_max_length=padding_offset + max_new_tokens,
    first_n=max_new_tokens,
    shuffle=False,
  )
  eval_labels = collections.defaultdict(list)
  eval_preds = collections.defaultdict(list)
  var_code = []
  source_label = []
  current_split = list(split_to_dataset)[0]
  with torch.no_grad():
    if debug_print:
      epoch_iterator = tqdm(eval_dataloader, desc="Test")
    else:
      epoch_iterator = eval_dataloader
    for step, inputs in enumerate(epoch_iterator):
      torch.cuda.empty_cache()
      b_s = inputs["input_ids"].shape[0]
      position_ids = {
        f"{prefix}position_ids": intervenable.model.prepare_inputs_for_generation(
          input_ids=inputs[f"{prefix}input_ids"],
          attention_mask=inputs[f"{prefix}attention_mask"],
        )["position_ids"]
        for prefix in ("", "source_")
      }
      inputs.update(position_ids)
      for key in inputs:
        if key in (
          "input_ids",
          "source_input_ids",
          "attention_mask",
          "source_attention_mask",
          "position_ids",
          "source_position_ids",
          "labels",
          "base_labels",
        ):
          inputs[key] = inputs[key].to(intervenable.model.device)
      if intervention_location_fn is not None:
        intervention_locations = intervention_location_fn(inputs, num_inv)
      else:
        intervention_locations = {
          "sources->base": (
            [
              [
                split_to_inv_locations[inputs["source_split"][i]][
                  "inv_position"
                ]
                for i in range(b_s)
              ]
            ]
            * num_inv,
            [
              [
                split_to_inv_locations[inputs["split"][i]]["inv_position"]
                for i in range(b_s)
              ]
            ]
            * num_inv,
          )
        }
      if inference_mode == "generate":
        base_outputs, counterfactual_outputs = intervenable.generate(
          {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
          },
          [
            {
              "input_ids": inputs["source_input_ids"],
              "attention_mask": inputs["source_attention_mask"],
              "position_ids": inputs["source_position_ids"],
            }
          ],
          intervention_locations,
          max_new_tokens=max_new_tokens,
          do_sample=False,
          intervene_on_prompt=True,
          pad_token_id=tokenizer.pad_token_id,
          output_original_output=True,
        )

        base_outputs = base_outputs[:, inputs["input_ids"].shape[1] :]
        counterfactual_outputs = counterfactual_outputs[
          :, inputs["input_ids"].shape[1] :
        ]
      elif inference_mode == "forward":
        base_outputs, counterfactual_outputs = intervenable(
          {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "position_ids": inputs["position_ids"],
          },
          [
            {
              "input_ids": inputs["source_input_ids"],
              "attention_mask": inputs["source_attention_mask"],
              "position_ids": inputs["source_position_ids"],
            }
          ],
          intervention_locations,
          output_original_output=True,
        )
        counterfactual_outputs = counterfactual_outputs.logits
        base_outputs = base_outputs.logits
      elif inference_mode == "force_decode":
        # Append the counterfactual label to the base input.
        # There are paddings both before the input and after the label.
        # P(y|x) = sum_i P(y_i|x_i, y_{j<i})
        force_decode_input_ids = torch.concat(
          [inputs["input_ids"], inputs["labels"]], dim=-1
        )
        force_decode_label = torch.concat(
          [torch.ones_like(inputs["input_ids"]) * -100, inputs["labels"]],
          dim=-1,
        )
        force_decode_attn_mask = torch.concat(
          [inputs["attention_mask"], torch.ones_like(inputs["labels"])], dim=-1
        )
        force_decode_position_ids = (
          intervenable.model.prepare_inputs_for_generation(
            input_ids=force_decode_input_ids,
            attention_mask=force_decode_attn_mask,
          )["position_ids"]
        )
        base_outputs, counterfactual_outputs = intervenable(
          {
            "input_ids": force_decode_input_ids,
            "attention_mask": force_decode_attn_mask,
            "position_ids": force_decode_position_ids,
            "labels": force_decode_label,
          },
          [
            {
              "input_ids": inputs["source_input_ids"],
              "attention_mask": inputs["source_attention_mask"],
              "position_ids": inputs["source_position_ids"],
            }
          ],
          # The appended labels do not change the intervention location, as
          # the intervention position is counted from the left.
          intervention_locations,
          output_original_output=True,
        )
        counterfactual_outputs = counterfactual_outputs.logits
        base_outputs = base_outputs.logits

      split_offset = [
        0,
        split_to_index[current_split][1] - step * eval_batch_size,
      ]
      while True:
        # Check if a split is finished or we are at the last batch.
        if (
          split_offset[1] >= 0 and split_offset[1] < eval_batch_size
        ) or step == len(epoch_iterator) - 1:
          if split_offset[1] > 0:
            eval_preds["base_outputs"].append(
              base_outputs[split_offset[0] : split_offset[1]]
            )
            eval_preds["inv_outputs"].append(
              counterfactual_outputs[split_offset[0] : split_offset[1]]
            )
            for label_type in ["base_labels", "labels"]:
              eval_labels[label_type].append(
                inputs[label_type][split_offset[0] : split_offset[1]]
              )
              eval_labels["base_outputs"].append(
                base_outputs[
                  split_offset[0] : split_offset[1], -max_new_tokens:
                ]
              )
            var_code.extend(
              inputs["var_code"][split_offset[0] : split_offset[1]]
              if "var_code" in inputs
              else inputs["label"][split_offset[0] : split_offset[1]]
            )
            source_label.extend(
              inputs["source_label"][split_offset[0] : split_offset[1]]
            )

          # Aggregate metrics
          # The compute metrics function computes the IIA
          eval_metrics = {
            label_type: compute_metrics_fn(
              keyed_eval_preds=eval_preds,
              eval_labels=eval_labels[label_type],
              last_n_tokens=max_new_tokens,
              pad_token_id=tokenizer.pad_token_id,
              extra_labels=eval_labels,
              eval_label_type=label_type,
            )
            for label_type in eval_labels
            if label_type.endswith("labels")
          }
          # print(f"\n\neval metrics \n{eval_metrics}")
          # eval metrics 
          # {'base_labels': {'base_outputs': {'accuracy': 1.0, 'token_accuracy': 1.0, 'class_0_accuracy': 0.4}, 'inv_outputs': {'accuracy': 0.8, 'token_accuracy': 0.8, 'class_0_accuracy': 0.3}, 'accuracy': 0.8}, 'labels': {'base_outputs': {'accuracy': 0.0, 'token_accuracy': 0.0, 'class_0_accuracy': 0.0}, 'inv_outputs': {'accuracy': 0.1, 'token_accuracy': 0.1, 'class_0_accuracy': 0.1}, 'accuracy': 0.1}}
          # 'source-In 2023 there-correct-test': {'base_labels': {'base_outputs': {'accuracy': 1.0, 'token_accuracy': 1.0, 'class_0_accuracy': 0.4}, 'inv_outputs': {'accuracy': 0.8, 'token_accuracy': 0.8, 'class_0_accuracy': 0.3}, 'accuracy': 0.8}, 'labels': {'base_outputs': {'accuracy': 0.0, 'token_accuracy': 0.0, 'class_0_accuracy': 0.0}, 'inv_outputs': {'accuracy': 0.1, 'token_accuracy': 0.1, 'class_0_accuracy': 0.1}, 'accuracy': 0.1}}


          inv_output_tokens = [
            extract_tokens_from_output(inference_mode, i, max_new_tokens, None)
            for i in eval_preds["inv_outputs"]
          ]

          inv_outputs = [
            tokenizer.batch_decode(
              remove_invalid_token_id(
                token_ids=i, pad_id=tokenizer.pad_token_id
              ),
              skip_special_tokens=True,
            )
            for i in inv_output_tokens
          ]
          # Merge inv_outputs into a single list
          inv_outputs = sum(inv_outputs, [])
          assert len(var_code) == len(inv_outputs), (
            f"len(var_code)={len(var_code)}, len(inv_outputs)={len(inv_outputs)}"
          )
          split_to_eval_metrics[current_split] = {
            "metrics": eval_metrics,
            "inv_outputs": inv_outputs,
            "inv_labels": tokenizer.batch_decode(
              remove_invalid_token_id(
                token_ids=inputs["labels"][:, :max_new_tokens],
                pad_id=tokenizer.pad_token_id,
              ),
              skip_special_tokens=True,
            ),
            "base_labels": tokenizer.batch_decode(
              remove_invalid_token_id(
                token_ids=inputs["base_labels"][:, :max_new_tokens],
                pad_id=tokenizer.pad_token_id,
              ),
              skip_special_tokens=True,
            ),
            "source_labels": source_label,
            "var_code": var_code,
          }
          print("\n", repr(current_split) + ":", eval_metrics)
          eval_preds = collections.defaultdict(list)
          eval_labels = collections.defaultdict(list)
          var_code = []
          source_label = []
          # Need to empty eval_preds to prevent OOM
          gc.collect()
          torch.cuda.empty_cache()
          # Check for termination condition.
          if len(split_to_eval_metrics) == len(split_to_dataset):
            break
          # Run the next split.
          current_split = list(split_to_dataset)[len(split_to_eval_metrics)]
          split_offset = [
            split_offset[1],
            split_to_index[current_split][1] - step * eval_batch_size,
          ]
        else:
          # Add the rest part of the split.
          # We could not compute the aggregated metrics now as we don't know if
          # there will be more examples from the same split in the next batch.
          eval_preds["base_outputs"].append(
            base_outputs[split_offset[0] : split_offset[1]]
          )
          eval_preds["inv_outputs"].append(
            counterfactual_outputs[split_offset[0] : split_offset[1]]
          )
          for label_type in ["base_labels", "labels"]:
            eval_labels[label_type].append(
              inputs[label_type][split_offset[0] : split_offset[1]]
            )
            eval_labels["base_outputs"].append(
              base_outputs[split_offset[0] : split_offset[1], -max_new_tokens:]
            )
          var_code.extend(
            inputs["var_code"][split_offset[0] : split_offset[1]]
            if "var_code" in inputs
            else inputs["label"][split_offset[0] : split_offset[1]]
          )
          source_label.extend(
            inputs["source_label"][split_offset[0] : split_offset[1]]
          )
          break

      # Debug logging.
      if debug_print and step < 3:
        # Check if the first entry is a 'pos' or 'h.pos'
        base_locs = (
          intervention_locations["sources->base"][1][0]
          if isinstance(
            intervention_locations["sources->base"][1][0][0][0], int
          )
          else intervention_locations["sources->base"][1][0][1]
        )
        source_locs = (
          intervention_locations["sources->base"][0][0]
          if isinstance(
            intervention_locations["sources->base"][0][0][0][0], int
          )
          else intervention_locations["sources->base"][0][0][1]
        )
        print("\nInputs:")
        print("Base:", inputs["input"][:3])
        print("Source:", inputs["source_input"][:3])
        print("Tokens to intervene:")
        print(
          "    Base:",
          tokenizer.batch_decode(
            [
              inputs["input_ids"][i][base_locs[i]]
              for i in range(len(inputs["split"]))
            ]
          ),
        )
        print(
          "    Source:",
          tokenizer.batch_decode(
            [
              inputs["source_input_ids"][i][source_locs[i]]
              for i in range(len(inputs["split"]))
            ]
          ),
        )

        print("Outputs:")
        for output_type, outputs in [
          ("base", base_outputs),
          ("counterfactual", counterfactual_outputs),
        ]:
          output_tokens = extract_tokens_from_output(
            inference_mode, outputs, max_new_tokens, prompt_tokens=None
          )
          if inference_mode == "generate" or inference_mode == "force_decode":
            output_text = tokenizer.batch_decode(output_tokens)
            if output_type == "base":
              base_output_text = output_text
          else:
            output_text = [
              "Top K:"
              + "|".join([tokenizer.decode(t) for t in output_tokens[i]])
              for i in range(len(output_tokens))
            ]
          print(f"{output_type.title()} Output:".rjust(22), output_text)

        base_label_text = []
        if inference_mode == "generate" or inference_mode == "force_decode":
          base_label_text = tokenizer.batch_decode(
            remove_invalid_token_id(
              token_ids=inputs["base_labels"][:, :max_new_tokens],
              pad_id=tokenizer.pad_token_id,
            ),
            skip_special_tokens=True,
          )
        else:
          base_label_text = [
            "|".join(tokenizer.batch_decode(label[:max_new_tokens]))
            for label in inputs["base_labels"]
          ]
        print("Labels:")
        print("Base Label:".rjust(22), base_label_text)
        print(
          "Counterfactual Label:".rjust(22),
          tokenizer.batch_decode(
            remove_invalid_token_id(
              token_ids=inputs["labels"][:, :max_new_tokens],
              pad_id=tokenizer.pad_token_id,
            ),
            skip_special_tokens=True,
          ),
        )
        if base_label_text != base_output_text and inference_mode == "generate":
          print("WARNING: Base outputs does not match base labels!")
  return split_to_eval_metrics


def compute_logits_metrics(
  keyed_eval_preds, eval_labels, pad_token_id, last_n_tokens=1, **kwargs
):
  """Computes logprob/loss/loss of mean logits of the eval_labels."""
  metrics = {}
  loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
  for key, eval_preds in keyed_eval_preds.items():
    logprob_agg = []
    loss_agg, loss_exp_agg = [], []
    pred_agg = []
    label_agg = []
    for eval_pred, eval_label in zip(eval_preds, eval_labels):
      eval_pred = eval_pred[:, -last_n_tokens - 1 : -1].contiguous()
      eval_label = eval_label[:, -last_n_tokens:].unsqueeze(-1).contiguous()
      assert len(eval_pred.shape) == 3
      assert eval_pred.shape[:2] == eval_label.shape[:2]
      # Index into prob with labels
      safe_labels = torch.maximum(eval_label, torch.zeros_like(eval_label))
      assert torch.max(safe_labels).tolist() < eval_pred.shape[-1]
      assert torch.min(safe_labels).tolist() >= 0
      probs = torch.nn.functional.softmax(eval_pred, dim=-1)
      logprob = torch.log(torch.gather(probs, -1, safe_labels))
      # Remove paddings
      is_padding = torch.logical_or(
        (safe_labels == pad_token_id), (safe_labels == 0)
      )
      logprob = torch.where(is_padding, torch.zeros_like(logprob), logprob)
      # Average the logprob of the whole sequence
      logprob = torch.sum(logprob, dim=-1) / torch.sum(~is_padding, dim=-1)
      logprob_agg.append(logprob.tolist())
      # Loss
      loss = loss_fn(
        eval_pred.view(-1, eval_pred.size(-1)), eval_label.view(-1)
      )
      loss = loss.view(eval_label.size(0), -1)
      loss_agg.extend(loss.mean(dim=-1).tolist())
      loss_exp_agg.extend((-loss.mean(dim=-1)).exp().tolist())
      pred_agg.append(eval_pred)
      label_agg.append(eval_label)
    # Compute the mean logits representation instead of the mean loss.
    # For a given key, all eval labels should be the same.
    loss_mean_repr = loss_fn(
      torch.mean(torch.cat(pred_agg, dim=0), dim=0).view(
        -1, eval_pred.size(-1)
      ),
      eval_label[0].view(-1),
    )
    metrics[key] = {
      "accuracy": -1,
      "token_accuracy": -1,
      "loss": np.mean(loss_agg).tolist(),
      "loss_exp": np.mean(loss_exp_agg).tolist(),
      "loss_max": np.max(loss_agg).tolist(),
      "loss_exp_max": np.max(loss_exp_agg).tolist(),
      "loss_min": np.min(loss_agg).tolist(),
      "loss_exp_min": np.min(loss_exp_agg).tolist(),
      "loss_mean_repr": loss_mean_repr[0].tolist(),
    }
  metrics["accuracy"] = metrics["inv_outputs"]["loss_mean_repr"]
  return metrics