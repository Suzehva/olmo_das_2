import abc
import numpy as np
import os
from functools import partial

from generation_utils import generate_batched, generate_distribution_batched
import torch

class TemplatedPrompts(metaclass=abc.ABCMeta):

  def __init__(self, templates, variable_name_to_vals):
    # variable_name_to_vals: {
    #    'var_name_0': ['val_0', 'val_1', ..., 'val_n'],
    #    ...
    #    'var_name_k': ['val_0', 'val_1', ..., 'val_m'],
    #  }
    self.templates = templates
    self.vars = variable_name_to_vals
    self.constraint_fn = None
    self.prompt_to_vars = self.get_all_prompts(constraint_fn=self.constraint_fn)

  @abc.abstractmethod
  def parse_variables_fn(self):
    pass

  @abc.abstractmethod
  def extract_prediction_fn(self):
    pass

  def get_all_prompts(self, constraint_fn):
    # total number of prompts <= |var_0| * |var_1| * ... * |var_n|
    templates = {}
    for template in self.templates:
      templates.update(
          get_all_prompts_partial({template: {}}, self.vars, constraint_fn))
    return templates

  def run_behavioral_test(self,
                          model,
                          tokenizer,
                          prompts,
                          batch_size=64,
                          max_new_tokens=1,
                          match_fn=None,
                          compare_logits=False,
                          multi_token_names=False,
                          top_k=100,
                          cache_dir=None,
                          split_type=None,
                          fold_id=None):
    assert compare_logits == False or (match_fn is None and max_new_tokens == 1)
    if compare_logits:
      if not multi_token_names:
        prompt_to_output, checked_prompts = eval_logits_correctness(
            self.parse_variables_fn(),
            model,
            tokenizer,
            prompts,
            top_k=top_k,
            batch_size=batch_size,
            cache_dir=cache_dir,
            split_type=split_type,
            fold_id=fold_id)
      else:
        prompt_to_output, checked_prompts = eval_logits_correctness_multitoken(
            self.parse_variables_fn(),
            model,
            tokenizer,
            prompts,
            batch_size=batch_size,
            cache_dir=cache_dir,
            split_type=split_type,
            fold_id=fold_id
        )
    else:
      prompt_to_output, checked_prompts = eval_output_correctness(
          self.extract_prediction_fn(),
          self.parse_variables_fn(),
          model,
          tokenizer,
          prompts,
          max_new_tokens,
          match_fn,
          batch_size,
          cache_dir=cache_dir,
          split_type=split_type,
          fold_id=fold_id)
    return prompt_to_output, checked_prompts

  def sample_correct_and_wrong_examples(self,
                                        model,
                                        tokenizer,
                                        num_samples,
                                        batch_size=64,
                                        max_new_tokens=1,
                                        match_fn=None,
                                        compare_logits=False,
                                        multi_token_names=False,
                                        top_k=100,
                                        cache_dir=None,
                                        split_type=None,
                                        fold_id=None):
    prompts = list(self.prompt_to_vars)
    print(f'Total #examples={len(prompts)}')
    prompt_to_output, checked_prompts = self.run_behavioral_test(
        model,
        tokenizer,
        prompts,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        match_fn=match_fn,
        compare_logits=compare_logits,
        multi_token_names=multi_token_names,
        top_k=top_k,
        cache_dir=cache_dir,
        split_type=split_type,
        fold_id=fold_id)
    # print(f"checked_prompts: {checked_prompts}")
    for category in ['correct', 'wrong']:
      filtered = []
      for p in checked_prompts[category]:
        out = prompt_to_output[p]
        # print(out)
        if hasattr(self, 'skip_output_fn') and self.skip_output_fn(out, tokenizer):
          # write in a txt file
          with open(f'skipped_output.txt', 'a', encoding='utf-8') as f:
            line = category + ' || ' + out + ' || ' + p.replace('\n', '<LINEBREAK>')
            # print(line)
            f.write(line+'\n')
          continue
        filtered.append(p)
      print(f"    #Skipped {category} examples={len(checked_prompts[category])-len(filtered)}  ")
      checked_prompts[category] = filtered

    # Random sample.
    rng = np.random.Generator(np.random.PCG64(seed=0))
    for k in checked_prompts:
      # what is k? k = 'correct' or 'wrong'
      rng.shuffle(checked_prompts[k])
      print(f'    #{k} examples={len(checked_prompts[k])}')
      print(f'Sampled    #{k} examples = min({num_samples}, {len(checked_prompts[k])}) = {min(num_samples, len(checked_prompts[k]))}')
      checked_prompts[k] = checked_prompts[k][:num_samples]

    # print(f"checked_prompts: {checked_prompts}")
    return prompt_to_output, checked_prompts


def get_model_id(model):
  return model.name_or_path.split('/')[-1]

def filter_single_token_names(names, tokenizer):
    """
    Returns a list of names that GPT-2 tokenizer encodes as exactly 1 token.
    Also returns the number of such names.
    """
    valid_names = []
    for name in names:
        token_ids = tokenizer.encode(name, add_special_tokens=False)
        # If it encodes to exactly 1 token, we keep it.
        if len(token_ids) == 1:
            valid_names.append(name)
    return valid_names, len(valid_names)

def filter_names_by_max_tokens(names, tokenizer, max_subtokens=2):
    """
    Keep only names that encode into at most 'max_subtokens' tokens.
    """
    valid_names = []
    for name in names:
        token_ids = tokenizer.encode(name, add_special_tokens=False)
        if len(token_ids) <= max_subtokens:
            valid_names.append(name)
    return valid_names, len(valid_names)

def get_all_prompts_partial(templates, vars, constraint_fn=None):
  # Example
  # ioi_template = ('Then, {name_a} and {name_b} were working at the {place}.'
  #                 ' {name_c} decided to give a {object} to')
  # prompts = get_all_prompts_partial(
  #     {ioi_template: {}},
  #     {'name_a': ['Alice', 'Bob', 'Carol'],
  #      'name_b': ['Alice', 'Bob', 'Carol'],
  #      'name_c': ['Alice', 'Bob', 'Carol'],
  #      'place': ['restaurant', 'office', 'school'],
  #      'object': ['cake', 'chips', 'drink']},
  #     ioi_prompt_constraints)
  if not vars:
    return templates
  # Fill the first var and recurse.
  var_name = list(vars)[0]
  assert all([var_name in t for t in templates])
  partial_filled_templates = {
      t.replace(f'{{{var_name}}}', v): dict(filled_vars, **{var_name: v})
      for t, filled_vars in templates.items() for v in vars[var_name] if
      constraint_fn is None or 
      constraint_fn(dict(filled_vars, **{var_name: v}))
  }
  return get_all_prompts_partial(
      partial_filled_templates,
      {k: v for k, v in vars.items() if k != var_name}, constraint_fn)


def eval_logits_correctness(parse_variables_fn,
                            model,
                            tokenizer,
                            prompts,
                            top_k=100,
                            batch_size=64,
                            margin=0,
                            cache_dir=None,
                            split_type=None,
                            fold_id=None):
  output_path = os.path.join(
      cache_dir,
      f'{get_model_id(model)}_prompt_to_top{top_k}_tokens_{split_type}_split_{fold_id}.pt') if cache_dir else None
  prompt_to_topk_tokens = {}
  if output_path and os.path.isfile(output_path):
    prompt_to_topk_tokens = torch.load(output_path)
  new_prompts = [p for p in prompts if p not in prompt_to_topk_tokens]
  if new_prompts:
    new_topk_tokens = generate_distribution_batched(model,
                                                    tokenizer,
                                                    new_prompts,
                                                    top_k=top_k,
                                                    batch_size=batch_size)
    new_prompt_to_topk_tokens = {
        p: new_topk_tokens[i] for i, p in enumerate(new_prompts)
    }
    prompt_to_topk_tokens.update(new_prompt_to_topk_tokens)
    if output_path:
      torch.save(prompt_to_topk_tokens, output_path)
  prompt_to_topk_tokens = {p: prompt_to_topk_tokens[p] for p in prompts}
  correct_examples, wrong_examples = [], []
  for p, dist in prompt_to_topk_tokens.items():
    vars = parse_variables_fn(p)
    correct_label, wrong_label = vars['positive_next_token'], vars[
        'negative_next_token']
    tok = [x[0] for x in dist]
    correct_prob = dist[tok.index(
        correct_label)][1] if correct_label in tok else 0
    wrong_prob = dist[tok.index(wrong_label)][1] if wrong_label in tok else 0
    if correct_prob - wrong_prob > margin:
      # print(p)
      # print(dist[tok.index(correct_label)])
      # print(dist[tok.index(wrong_label)])
      # print(correct_prob, wrong_prob)
      correct_examples.append(p)
    else:
      wrong_examples.append(p)
  return prompt_to_topk_tokens, {
      'correct': correct_examples,
      'wrong': wrong_examples
  }


def eval_logits_correctness_multitoken(parse_variables_fn,
                                       model,
                                       tokenizer,
                                       prompts,
                                       batch_size=64,
                                       margin=0.0,
                                       cache_dir=None,
                                       split_type=None,
                                       fold_id=None):
    """
    Checks multi-token correctness by comparing average log probs of 'correct_label' vs. 'wrong_label'.
    """
    def get_avg_log_prob_of_continuation(prompt_text, continuation):
      """
      Returns the average log prob of `continuation` given `prompt_text`.
      Uses model(..., labels=...) so that only the tokens of `continuation`
      contribute to the cross-entropy loss.
      """
      with torch.no_grad():
        # 1) Encode prompt & continuation
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        cont_ids   = tokenizer.encode(continuation, add_special_tokens=False)
        full_ids   = prompt_ids + cont_ids
        # 2) Convert to a single batch
        input_tensor = torch.tensor([full_ids], dtype=torch.long, device=model.device)
        # Labels: mark prompt tokens with -100, continuation tokens with actual IDs
        labels = torch.full_like(input_tensor, fill_value=-100)
        cont_start = len(prompt_ids)
        labels[0, cont_start:] = input_tensor[0, cont_start:]
        # 3) Forward pass => model(...).loss is average cross-entropy over continuation
        outputs = model(input_ids=input_tensor, labels=labels)
        # outputs.loss is a scalar = average negative log-prob of the continuation tokens
        avg_neg_log_prob = outputs.loss.item()  # > 0
        avg_log_prob     = -avg_neg_log_prob     # average log probability
        return avg_log_prob
    
    # cache
    output_path = (os.path.join(
        cache_dir,
        f'{get_model_id(model)}_prompt_to_multitoken_{split_type}_split_{fold_id}.pt') 
        if cache_dir else None)
    prompt_to_topk_tokens = {}
    if output_path and os.path.isfile(output_path):
        prompt_to_topk_tokens = torch.load(output_path)
    new_prompts = [p for p in prompts if p not in prompt_to_topk_tokens]

    # Compute for new prompts
    if new_prompts:
      for p in new_prompts:
        vars = parse_variables_fn(p)
        # print(f"vars: {vars}")
        correct_label = vars['positive_next_token']  # e.g. multi-token name
        wrong_label   = vars['negative_next_token']

        # Truncate both labels to n tokens
        correct_ids = tokenizer.encode(correct_label, add_special_tokens=False)
        wrong_ids   = tokenizer.encode(wrong_label, add_special_tokens=False)
        n = min(len(correct_ids), len(wrong_ids))
        correct_ids_trunc = correct_ids[:n]
        wrong_ids_trunc   = wrong_ids[:n]
        correct_label_trunc = tokenizer.decode(correct_ids_trunc, skip_special_tokens=True)
        wrong_label_trunc   = tokenizer.decode(wrong_ids_trunc, skip_special_tokens=True)
        correct_lp = get_avg_log_prob_of_continuation(p, correct_label_trunc)
        wrong_lp   = get_avg_log_prob_of_continuation(p, wrong_label_trunc)

        prompt_to_topk_tokens[p] = [
            (correct_label_trunc, correct_lp),
            (wrong_label_trunc,  wrong_lp)
        ]
      if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save(prompt_to_topk_tokens, output_path)
    # Final pass over *all* prompts => correct or wrong
    correct_examples = []
    wrong_examples   = []

    for p in prompts:
        label_probs = prompt_to_topk_tokens[p]  # e.g. [(labelA, logprobA), (labelB, logprobB)]
        correct_lp = label_probs[0][1]
        wrong_lp   = label_probs[1][1]
        if (correct_lp - wrong_lp) > margin:
            correct_examples.append(p)
        else:
            wrong_examples.append(p)

    return prompt_to_topk_tokens, {
        'correct': correct_examples,
        'wrong': wrong_examples
    }


def eval_output_correctness(extract_prediction_fn,
                            parse_variables_fn,
                            model,
                            tokenizer,
                            prompts,
                            max_new_tokens,
                            match_fn,
                            batch_size=64,
                            cache_dir=None,
                            split_type=None,
                            fold_id=None):
  if match_fn is None:
    print(f"Note: match_fn is None, using default match_fn")
    match_fn = lambda vars, pred: vars['label'] == pred
  output_path = os.path.join(
      cache_dir,
      f'{get_model_id(model)}_prompt_to_output_{split_type}_split_{fold_id}.pt') if cache_dir else None
  prompt_to_output = {}
  if output_path and os.path.isfile(output_path):
    prompt_to_output = torch.load(output_path)
    print(f'...Load {len(prompt_to_output)} examples from the cache.')
  new_prompts = [p for p in prompts if p not in prompt_to_output]
  print(f'...Found {len(prompts) - len(new_prompts)} prompts in the cache.')
  if new_prompts:
    new_prompt_to_output = generate_batched(model,
                                            tokenizer,
                                            new_prompts,
                                            max_new_tokens=max_new_tokens,
                                            batch_size=batch_size)
    # print(f"new_prompt_to_output: {new_prompt_to_output}")
    # new_prompt_to_output = {
    #     p:
    #     out[len(tokenizer.decode(tokenizer.encode(p), skip_special_tokens=True)):] 
    #     for p, out in new_prompt_to_output.items()
    # }
    # print(f"new_prompt_to_output: {new_prompt_to_output}")

    prompt_to_output.update(new_prompt_to_output)
    # print(f"prompt_to_output: {prompt_to_output}")
    if output_path:
      os.makedirs(os.path.dirname(output_path), exist_ok=True)
      torch.save(prompt_to_output, output_path)
      
  prompt_to_output = {p: prompt_to_output[p] for p in prompts}
  correct_examples, wrong_examples = [], []

  for p, out in prompt_to_output.items():
    prediction = extract_prediction_fn(out)
    label = parse_variables_fn(p)['label']
    vars = parse_variables_fn(p)
    # print(f'out={out}, prediction={prediction}, label={label}')
    # print(f"label: {label}")
    # print(f"prediction: {prediction}")
    if match_fn(vars, prediction):
      correct_examples.append(p)
    else:
      # print(repr(p))
      # print(repr(out))
      # print(prediction)
      # print(label)
      wrong_examples.append(p)
  return prompt_to_output, {
      'correct': correct_examples,
      'wrong': wrong_examples
  }
