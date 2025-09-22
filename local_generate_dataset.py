import collections
import json
import random
import os
import copy
import numpy as np

from ioi_templated_prompts import IoiTemplatedPrompts
#from pricetag_templated_prompts import PricetagTemplatedPrompts
#from ravel_templated_prompts import RavelTemplatedPrompts
#from ravel_qa_templated_prompts import RavelQATemplatedPrompts


def deterministic_shuffle(x, inplace=True, seed=0):
  if not inplace:
    x = copy.deepcopy(x)
  rng = np.random.Generator(np.random.PCG64(seed=seed))
  rng.shuffle(x)
  return None if inplace else x


def get_templated_prompt_class(class_name):
  if class_name == 'IoiTemplatedPrompts':
    return IoiTemplatedPrompts
  elif class_name == 'PricetagTemplatedPrompts':
    return PricetagTemplatedPrompts
  elif class_name == 'RavelTemplatedPrompts':
    return RavelTemplatedPrompts
  elif class_name == 'RavelQATemplatedPrompts':
    return RavelQATemplatedPrompts
  else:
    raise ValueError(f'Unknown class name: {class_name}')


def generate_templated_prompt_dataset(model, tokenizer, config):
  split_to_templates = config['split_to_templates']
  split_to_vars = config['split_to_vars']
  templated_prompt_class = get_templated_prompt_class(
      config['templated_prompt_class'])
  split_to_examples = {}

  for split, templates in split_to_templates.items():
    # split is 'train', 'val', 'test'
    prompts = templated_prompt_class(templates, split_to_vars[split])
    print(f"\n# {split} split raw prompts = {len(prompts.prompt_to_vars)} ->")

    # Downsample to max_prompts if we have more
    max_prompts = config['max_prompts']
    all_keys = list(prompts.prompt_to_vars.keys())
    if len(all_keys) > max_prompts:
      deterministic_shuffle(all_keys, inplace=True)
      selected = all_keys[:max_prompts]
      # Build a new dictionary
      prompts.prompt_to_vars = {k: prompts.prompt_to_vars[k] for k in selected}
    # Final # of prompts after downsample
    print(f"# {split} split after downsample = {len(prompts.prompt_to_vars)}")
    print(f"----------- start sample correct and wrong examples -----------")
    # Sample “correct” and “wrong” examples
    this_num_samples = config['num_samples'][split] if isinstance(
        config['num_samples'], dict) else config['num_samples']

    os.makedirs(config['cache_dir'], exist_ok=True)
    prompt_to_output, checked_prompts = prompts.sample_correct_and_wrong_examples(
        model,
        tokenizer,
        num_samples=this_num_samples,
        batch_size=config['batch_size'],
        max_new_tokens=config['max_new_tokens'],
        match_fn=config['match_fn'],
        compare_logits=config['compare_logits'],
        multi_token_names=config['multi_token_names'],
        top_k=config['top_k'],
        cache_dir=config['cache_dir'],
        split_type=config['split_type'],
        fold_id=config['fold_id'])
    split_to_examples[split] = checked_prompts
  return split_to_examples


# Example
#
# from ravel_templated_prompts import ravel_match_fn, RavelTemplatedPrompts
#
#
# var_to_freq = json.load(open(os.path.join('abstraction/data/city_counts_v4_dolma-v1_7_llama.json')))
# freq_sorted_vars = sorted(var_to_freq, key=var_to_freq.get, reverse=True)
#
# templates = [
#     '[{"city": "Kuala Lumpur", "country": "Malaysia"}, {"city": "{city}", "country": "'
# ]
#
# config = {
#     'split_to_templates': {
#         'train': templates,
#         'val': templates,
#         'test': templates,
#     },
#     'split_to_vars': {
#         'train': {'city': freq_sorted_vars[20:500] + freq_sorted_vars[1000:1500]},
#         'val': {'city': freq_sorted_vars[500:1000]},
#         'test': {'city': freq_sorted_vars[:20] + freq_sorted_vars[1500:]},
#     },
#     'templated_prompt_class': 'RavelTemplatedPrompts',
#     'num_samples': 512,
#     'batch_size': 64,
#     'max_new_tokens': 8,
#     'match_fn': ravel_match_fn,
#     'compare_logits': False,
#     'top_k': None,
#     'cache_dir': os.path.join(SCR_DATA_DIR, 'ravel_outputs'),
# }
#
# split_to_dataset = generate_templated_prompt_dataset(model, tokenizer, config)
