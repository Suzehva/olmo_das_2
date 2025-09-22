import numpy as np
import os

import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer
try:
  import plotly.express as px
except:
  pass


def compute_per_token_loss(model, encoded_inputs, labels, temperature=None):
  loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
  with torch.no_grad():
    # position_ids is required if there are paddings.
    position_ids = encoded_inputs.attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(encoded_inputs.attention_mask == 0, 1)
    outputs = model(**encoded_inputs, position_ids=position_ids, labels=labels)
    shift_logits = outputs.logits[:, :-1, :].contiguous()
    if temperature is not None:
      shift_logits = shift_logits / temperature
    labels = labels[:, 1:].unsqueeze(-1).contiguous().to(torch.int64)
    loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)),
                   labels.view(-1))
    loss = loss.view(labels.size(0), -1)
    return loss


def compute_per_token_loss_batched(model,
                                   encoded_inputs,
                                   labels,
                                   temperature=None,
                                   batch_size=64):
  loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
  loss_agg = []
  for b_i in tqdm(range(0, len(labels), batch_size)):
    with torch.no_grad():
      # position_ids is required if there are paddings.
      input_ids = encoded_inputs.input_ids[b_i:b_i + batch_size]
      attention_mask = encoded_inputs.attention_mask[b_i:b_i + batch_size]
      batch_labels = labels[b_i:b_i + batch_size]
      position_ids = attention_mask.long().cumsum(-1) - 1
      position_ids.masked_fill_(attention_mask == 0, 1)
      outputs = model(input_ids=input_ids,
                      attention_mask=attention_mask,
                      position_ids=position_ids,
                      labels=batch_labels)
      shift_logits = outputs.logits[:, :-1, :].contiguous()
      if temperature is not None:
        shift_logits = shift_logits / temperature
      batch_labels = batch_labels[:,
                                  1:].unsqueeze(-1).contiguous().to(torch.int64)
      loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)),
                     batch_labels.view(-1))
      loss = loss.view(batch_labels.size(0), -1)
      loss_agg.extend(loss.tolist())
  return loss_agg


def compute_per_token_probability(model,
                                  encoded_inputs,
                                  labels,
                                  temperature=None):
  with torch.no_grad():
    # position_ids is required if there are paddings.
    position_ids = encoded_inputs.attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(encoded_inputs.attention_mask == 0, 1)
    attention_mask = encoded_inputs.attention_mask
    # attention_mask = encoded_inputs.attention_mask.float().masked_fill(encoded_inputs.attention_mask == 0, -torch.inf).to(model.dtype)
    # for debugging:
    # print(attention_mask.shape)
    # print(encoded_inputs.input_ids.shape)
    # print(torch.ne(encoded_inputs.input_ids, 128009).sum())
    outputs = model(input_ids=encoded_inputs.input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids, labels=labels)

    shift_logits = outputs.logits[:, :-1, :].contiguous()
    if temperature is not None:
      shift_logits = shift_logits / temperature
    full_dist = torch.nn.functional.softmax(shift_logits, dim=-1)
    labels = labels[:, 1:].unsqueeze(-1).contiguous().to(torch.int64)
    # Index into prob with labels
    safe_labels = torch.maximum(labels, torch.zeros_like(labels))
    assert torch.max(safe_labels).tolist() < full_dist.shape[-1]
    assert torch.min(safe_labels).tolist() >= 0
    label_prob = torch.gather(full_dist, -1, safe_labels)
    label_prob = torch.where(labels >= 0, label_prob,
                             torch.ones_like(label_prob) * -1)
    return label_prob, full_dist


def compute_confidence_score(
    model,
    tokenizer,
    prompt_batch,
    output_batch,
    score_fn=None,
    max_length=128,
    batch_size=32,
    temperature=None,
):
  assert len(prompt_batch) == len(output_batch)
  input_batch = [
      prompt_batch[i] + output_batch[i] for i in range(len(prompt_batch))
  ]

  all_scores = []
  for i in tqdm(range(0, len(input_batch), batch_size)):
    encoded_inputs = tokenizer(input_batch[i:i + batch_size],
                               return_tensors='pt',
                               max_length=max_length,
                               truncation=True,
                               padding='max_length').to(model.device)
    encoded_outputs = tokenizer(output_batch[i:i + batch_size],
                                return_tensors='pt',
                                max_length=max_length,
                                truncation=True,
                                padding='max_length').to(model.device)
    labels = encoded_inputs['input_ids'].clone()
    if (tokenizer.bos_token_id is not None and
        tokenizer.bos_token_id != tokenizer.pad_token_id):
      # Tokenizer with a BOS token.
      prompt_mask = ~torch.logical_and(
          encoded_outputs.attention_mask.to(torch.bool),
          encoded_outputs.input_ids != tokenizer.bos_token_id)
    else:
      prompt_mask = ~encoded_outputs.attention_mask.to(torch.bool)
    labels = torch.where(prompt_mask, torch.ones_like(labels) * -100, labels)
    label_prob, full_dist = compute_per_token_probability(
        model, encoded_inputs, labels, temperature)
    # Mask out prompt and padding tokens.
    full_dist = torch.where(prompt_mask[:, 1:].unsqueeze(-1),
                            torch.zeros_like(full_dist), full_dist)
    if score_fn is None:
      # Take the probability of the label token of the last step.
      scores = label_prob[:, -1, 0].tolist()
    else:
      scores = score_fn(full_dist)
    all_scores.extend(scores)

  return all_scores


def pool_confidence_score(tokenizer,
                          prob,
                          next_n_tokens=16,
                          kept_token_ids=None,
                          mode='mean'):
  epsilon = 1e-4
  pred_tokens = torch.argmax(prob, dim=-1)
  top_token_conf = torch.max(prob, dim=-1)[0]
  confidence_score = []
  for b_i in range(len(prob)):
    # 1) Find how many tokens are effectively "non-zero"
    num_pad = torch.sum(torch.max(prob[b_i], dim=-1)[0] < epsilon)
    kept_prob = top_token_conf[b_i]
    if kept_token_ids is not None:
      # Possibly restrict to certain token IDs
      mask_any = torch.stack([
          (pred_tokens[b_i] == tid) for tid in kept_token_ids
      ])
      mask_any = torch.any(mask_any, dim=0)
      kept_prob = torch.where(mask_any, kept_prob, torch.zeros_like(kept_prob))
    # 2) Slice the first N tokens after skipping padding
    kept_prob = kept_prob[num_pad:num_pad + next_n_tokens]
    # 3) Avoid dividing by zero:
    nonzero = kept_prob[kept_prob > epsilon]
    if len(nonzero) == 0:
      # no valid tokens => fallback to 0.0 or some small number
      confidence_score.append(0.0)
      continue
    if mode == 'mean':
      nonzero = torch.log(nonzero)
      confidence_score.append((nonzero.mean().exp().item()))
      # confidence_score.append((nonzero.mean().item()))
    elif mode == 'max':
      confidence_score.append((nonzero.max().item()))
    elif mode == 'min':
      confidence_score.append((nonzero.min().item()))
    else:
      raise ValueError(f"Unknown mode: {mode}")
  return confidence_score

def print_output_length(tokenizer, test_output_batch):
  # `test_output_batch` a list of raw strings from the model
  # model output stored in a dict prompt_to_output
  all_lengths = []
  for out in test_output_batch:
    # Tokenize the model's *output* (not the prompt+output).
    out_ids = tokenizer.encode(out, add_special_tokens=False)
    all_lengths.append(len(out_ids))
  avg_len = np.mean(all_lengths)
  max_len = np.max(all_lengths)
  print(f"Average # tokens in model output: {avg_len:.2f} (max = {max_len})")
  return avg_len, max_len


def find_best_nextN_for_confidence_score(
    model,
    tokenizer,
    val_prompt_batch,
    val_output_batch,
    val_labels,  # boolean or 0/1 ground truth
    max_length=128,
    N_candidates=range(1, 11),
    mode='mean',
    data_config=None,
    output_dir=None):
  """
  1) For each N in N_candidates:
        compute confidence scores
        compute AUC–ROC
  2) Return the best N and the associated AUC–ROC.
  """ 
  best_auc = 0.0
  best_n = None
  results = []  # to store (N, AUC) pairs
  for N in N_candidates:
    # print(f"Computing AUC–ROC for N={N}")
    # Compute confidence scores using next_n_tokens=N
    confidence_score = compute_confidence_score(
        model,
        tokenizer,
        prompt_batch=val_prompt_batch,
        output_batch=val_output_batch,
        score_fn=lambda probs: pool_confidence_score(
            tokenizer, probs, next_n_tokens=N, mode=mode),
        max_length=max_length)
    # Evaluate with AUC–ROC
    # val_labels is 0/1, `confidence_score` is continuous
    auc_roc = roc_auc_score(val_labels, confidence_score)
    print(f"N={N} => AUC–ROC={auc_roc:.4f}")
    results.append((N, auc_roc))
    if auc_roc > best_auc:
      best_auc = auc_roc
      best_n = N
  model_name, task_name, split_type, fold_id = data_config[
      'model_name'], data_config['task_name'], data_config[
          'split_type'], data_config['fold_id']
  # plot
  df_results = pd.DataFrame(results, columns=['N', 'AUC_ROC'])
  # save the dataframe
  os.makedirs(os.path.join(output_dir, f'{task_name}_task'), exist_ok=True)
  df_results.to_csv(os.path.join(
      output_dir, f'{task_name}_task',
      f'{task_name}_task_{model_name}_{split_type}_split_{fold_id}_auc_roc_vs_n.csv'
  ),
                    index=False)
  fig = px.line(
      df_results,
      x='N',
      y='AUC_ROC',
      markers=True,
      title=f'AUC–ROC vs. Number of Tokens (N) for {split_type}_split_{fold_id}'
  )
  fig.update_layout(
      xaxis_title="Number of Tokens (N)",
      yaxis_title="AUC–ROC",
      yaxis_range=[0, 1.0]  # if you want y from 0 to 1
  )
  fig.show()
  # save the figure
  fig.write_image(
      os.path.join(
          output_dir, f'{task_name}_task',
          f'{task_name}_task_{model_name}_{split_type}_split_{fold_id}_auc_roc_vs_n.png'
      ))
  return best_n, best_auc


def compute_expected_calibration_error(confidence_score, label, n_bins=10):
  assert (np.max(confidence_score) <= 1 and np.min(confidence_score) >= 0)
  # Adapted from https://github.com/scikit-learn/scikit-learn/blob/6e9039160/sklearn/calibration.py#L927
  y_prob = confidence_score
  y_true = label
  bins = np.linspace(0.0, 1.0, n_bins + 1)
  binids = np.searchsorted(bins[1:-1], y_prob)
  bin_sums = np.bincount(binids, weights=y_prob, minlength=len(bins))
  bin_true = np.bincount(binids, weights=y_true, minlength=len(bins))
  bin_total = np.bincount(binids, minlength=len(bins))
  prob_true = (bin_true / np.maximum(bin_total, 1.0))[:-1]
  prob_pred = (bin_sums / np.maximum(bin_total, 1.0))[:-1]
  bin_counts = np.bincount(binids, minlength=len(bins))[:-1]
  bin_weights = bin_counts / bin_counts.sum()
  print(bin_counts)
  ece = (np.abs(prob_true - prob_pred) * bin_weights).sum()
  return ece, prob_true, prob_pred, bin_weights


def plot_ece(prob_true, prob_pred, num_bins, title, bin_weights=None):
  if bin_weights is not None:
    bin_weights = (bin_weights - bin_weights.min()) / (bin_weights.max() -
                                                       bin_weights.min())
  df = pd.DataFrame([{
      'conf_bucket':
          i,
      'accuracy':
          prob_true[i].tolist(),
      'bin_weights':
          bin_weights[int(c * num_bins)] if bin_weights is not None else 1
  } for i, c in enumerate(prob_pred)])
  sns.set_theme()
  ax = sns.barplot(df,
                   x='conf_bucket',
                   y='accuracy',
                   hue='bin_weights',
                   width=1,
                   palette='Blues',
                   hue_norm=(-0.2, 1.2))
  # draw a diagonal line
  ax.plot([-0.5, num_bins - 0.5], [0, 1], 'k--')
  ax.set_ylim(0, 1.0)
  ax.set_xlim(-0.5, num_bins - 0.5)
  ax.set_ylabel('P(correct)')
  ax.set_xlabel('P(answer)')
  ax.set_xticklabels([
      f'[{x:.1f}, {x+1/num_bins:.1f}]'
      for x in np.arange(0.0, 1.0, 1 / num_bins)
  ],
                     rotation=90)
  ax.set_aspect(num_bins)
  ax.set_title(title)
  ax.legend().remove()


def make_score_fn(tokenizer, N, mode):
    """
    Returns a function that accepts `probs` and calls
    pool_confidence_score(tokenizer, probs, next_n_tokens=N, mode=mode).
    """
    def score_fn(probs):
        return pool_confidence_score(tokenizer, probs, next_n_tokens=N, mode=mode)
    return score_fn
