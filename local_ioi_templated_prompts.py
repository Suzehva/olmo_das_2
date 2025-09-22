import re

from templated_prompts import TemplatedPrompts

@staticmethod

def ioi_prompt_constraints(vars):
  diff_s_io, same_s = True, True
  if 'name_a' in vars and 'name_b' in vars:
    diff_s_io = vars['name_a'] != vars['name_b']
    # tokenized_name_a, tokenized_name_b = tokenizer.encode(vars['name_a']), tokenizer.encode(vars['name_b'])
    # diff_2tok = tokenized_name_a[:2] != tokenized_name_b[:2]
    if 'name_c' in vars:
      same_s = vars['name_a'] == vars['name_c'] or vars['name_b'] == vars['name_c']
  return diff_s_io and same_s # and diff_2tok

def ioi_match_fn(vars, pred):
  # print(f'vars: {vars}')
  # print(f'pred: {pred}')
  return pred == vars['positive_next_token'].replace(' ', '')

def get_position_label(vars, parsed_output):
  if parsed_output == vars['first_name']:
    return 'first'
  elif parsed_output == vars['second_name']:
    return 'second'
  else:
    return '<invalid>'

def ioi_get_inv_label(target_var=None, base_vars=None, source_vars=None):
  '''
  Get the intervention label.
  '''

  if 'position_label' in source_vars:
    position_label_source = source_vars['position_label']
    if position_label_source == '<invalid>':
      # can potentially be met
      # print()
      # print(f'source_vars: {source_vars}')
      # print(f'base_vars: {base_vars}')
      return '<EMPTY>'
    else:
      return base_vars[f'{position_label_source}_name']
  else:
    # print('Invalid position_label')
    # print(f'source_vars: {source_vars}')
    # print(f'base_vars: {base_vars}')
    raise ValueError(f'Invalid position_label')

def ioi_filter_inv_example(target_var=None, base_vars=None, source_vars=None):
  '''
  Whether to keep an example for an intervention.
  '''
  if source_vars['position_label'] == base_vars['position_label']:
    return False
  return True

def ioi_parse_variables(prompt, vars_dict):
  name_template_pattern = re.compile(r' ([A-Z][a-z]+) and ([A-Z][a-z]+)')
  name_matches = re.search(name_template_pattern, prompt)
  object_matches = re.search(r'|'.join(vars_dict['object']), prompt)
  place_matches = re.search(r'|'.join(vars_dict['place']), prompt)
  vars = {}
  if name_matches is None:
    raise ValueError(f'Invalid template: {prompt}')
  if object_matches and place_matches:
    vars['object'] = object_matches.group()
    vars['place'] = place_matches.group()
  if prompt.count(name_matches[1] + ' ') == 2:
    vars['first_name'] = name_matches[1]
    vars['second_name'] = name_matches[2]
    vars['subject'] = name_matches[1]
    vars['indirect_object'] = name_matches[2]
    vars['order'] = 'ABA'
  elif prompt.count(name_matches[2] + ' ') == 2:
    vars['first_name'] = name_matches[1]
    vars['second_name'] = name_matches[2]
    vars['subject'] = name_matches[2]
    vars['indirect_object'] = name_matches[1]
    vars['order'] = 'BAA'
  else:
    print(prompt)
    raise ValueError('Invalid template')
  vars['label'] = vars['indirect_object'] # will be overwritten by the parsed model output
  vars['positive_next_token'] = ' ' + vars['indirect_object']
  vars['negative_next_token'] = ' ' + vars['subject']
  vars['position_label'] = '<EMPTY>'
  return vars

def extract_prediction(out):
  
  # out = out.strip().replace('\n', ' ').replace('"', '').replace(".", '')
  # name_pred_pattern = re.compile(r'is[^\w]+([A-Z][a-z]+)"?|^\s([A-Z][a-z]+)$|^([A-Z][a-z]+)$')
  # name_match = re.search(name_pred_pattern, out)
  # if not name_match:
  #   return '<EMPTY>'
  # return re.search(r'[A-Z][a-z]+', name_match[0])[0]
  out = re.sub(r"Based on the story, "
               "|A [a-z]+ question!"
               "|A [a-z]+ story!"
               "|Based on the context, "
              #  "|it( seems| is likely|\'s likely) that "
               "|\\n"
               ,
               "", 
               out)

  pattern = re.compile(
        r"""
         #          Pattern 1) 
        (?:.*?
          (?P<name1>[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)?)  # e.g. "Karo", "Jean Claude"
          \s*
          (?:receives|gets|(will|might)\s+receive
          |(will|might)\s+get|receive|get|is\s+the\s+one\s+receiving|is\s+the\s+one\s+who\s+(will\s+receive|receives|gets)
          |is\s+likely\s+to\s+receive
          )  # some verb phrase
          .*?
        )
        | # -- OR -- Pattern 2.1) "it seems that Murad is likely to give the bone to Seanmichael"
        
        (?:will\s+give|gives|is\s+giving)   # the verb phrase
        \s+(?:a\s+|the\s+)?[a-z]+\s+to\s+
        (?P<name2_1>[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)?)
        # .*$
        | # -- OR -- Pattern 2.2) "the bone will be given to Anuja"
        
        (?:will\s+be\s+given\s+to.*\s+
          (?P<name2_2>[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)?)
          .*?
        )
        | # -- OR -- Pattern 3) "X is the most likely recipient"
        
        (?:  # We look for "is the (some optional words) likely recipient"
          (?P<name3>[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)?) # the name
          \s+.*?
          is\s+          # "is"
          (?:the\s+)?    # optionally "the"
          (?:\w+\s+){0,3}?   # optional short filler words (like "most", "more", "likely")
          recipient      # the word "recipient"
        )
        | # -- OR -- Pattern 4.1) A line that starts with "Answer: SomeName"
        
        (I\s*would\s*answer:|Answer:|answer\s*is)\s*(?P<name4_1>[A-Z][A-Za-z]+)
        (!|.)+
        | # -- OR -- Pattern 4.2) A single capitalized word line with optional punctuation:
        ^
        (?P<name4_2>[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)?)
        (!|.)*

        """,
        re.MULTILINE | re.VERBOSE
    )
  
  other_subjects = ['student', 'patient', 'dog', 'pet', 'customer']
  # the students at the school|the patients at the hospital
  # a customer
  # the dog|a pet
  # someone|someone else
  # Based on the story, it is likely that the dog, Carmeron, receives the bone in the end.
  match = pattern.search(out)
  if not match:
      match_other_subjects=False
      for s in other_subjects:
        if s in out:
          match_other_subjects=True
          break
      # if not match_other_subjects:
      #   print(out)
      return "<EMPTY>"
  # One of the three named groups might have matched
  if match.groupdict().get('name4_2'):
    # print(f'name4_2: {match.groupdict().get("name4_2")}')
    return match.groupdict().get('name4_2')
  if match.groupdict().get('name4_1'):
    # print(f'name4_1: {match.groupdict().get("name4_1")}')
    return match.groupdict().get('name4_1')
  name_candidate = (
    match.groupdict().get('name1') or
    match.groupdict().get('name2_1') or
    match.groupdict().get('name2_2') or 
    match.groupdict().get('name3') 
  )
  # print which condition got matched
  # if match.groupdict().get('name1'):
  #   print(f'name1: {match.groupdict().get("name1")}')
  # if match.groupdict().get('name2_1'):
  #   print(f'name2_1: {match.groupdict().get("name2_1")}')
  # if match.groupdict().get('name2_2'):
  #   print(f'name2_2: {match.groupdict().get("name2_2")}')
  # if match.groupdict().get('name3'):
  #   print(f'name3: {match.groupdict().get("name3")}')

  if not name_candidate:
      return '<EMPTY>'
  return name_candidate

class IoiTemplatedPrompts(TemplatedPrompts):

  def __init__(self, templates, variable_name_to_vals):
    self.templates = templates
    self.vars = variable_name_to_vals
    self.constraint_fn = ioi_prompt_constraints
    self.prompt_to_vars = self.get_all_prompts(constraint_fn=self.constraint_fn)

    self.name_template_pattern = re.compile(r' ([A-Z][a-z]+) and ([A-Z][a-z]+)')
    self.name_pred_pattern = re.compile(
        r'is[^\w]+([A-Z][a-z]+)"?|^\s([A-Z][a-z]+)$|^([A-Z][a-z]+)$')


  def parse_variables_fn(self):
    def parse_variables(prompt):
      return ioi_parse_variables(prompt, self.vars)
    return parse_variables
  

  @staticmethod
  def ioi_parse_variables(prompt, vars_dict):
    name_template_pattern = re.compile(r' ([A-Z][a-z]+) and ([A-Z][a-z]+)')
    name_matches = re.search(name_template_pattern, prompt)
    object_matches = re.search(r'|'.join(vars_dict['object']), prompt)
    place_matches = re.search(r'|'.join(vars_dict['place']), prompt)
    vars = {}
    if name_matches is None:
      raise ValueError(f'Invalid template: {prompt}')
    if object_matches and place_matches:
      vars['object'] = object_matches.group()
      vars['place'] = place_matches.group()
    if prompt.count(name_matches[1] + ' ') == 2:
      vars['first_name'] = name_matches[1]
      vars['second_name'] = name_matches[2]
      vars['subject'] = name_matches[1]
      vars['indirect_object'] = name_matches[2]
      vars['order'] = 'ABA'
    elif prompt.count(name_matches[2] + ' ') == 2:
      vars['first_name'] = name_matches[1]
      vars['second_name'] = name_matches[2]
      vars['subject'] = name_matches[2]
      vars['indirect_object'] = name_matches[1]
      vars['order'] = 'BAA'
    else:
      print(prompt)
      raise ValueError('Invalid template')
    vars['label'] = vars['indirect_object'] # will be overwritten by the parsed model output
    vars['positive_next_token'] = ' ' + vars['indirect_object']
    vars['negative_next_token'] = ' ' + vars['subject']
    vars['position_label'] = '<EMPTY>'
    return vars

  def extract_prediction_fn(self):
    return extract_prediction

  def skip_output_fn(self, out, tokenizer):
    fail_to_extract = False
    if extract_prediction(out) == '<EMPTY>':
      fail_to_extract = True
    return fail_to_extract

    # diff_2tok = True # filter out names that have the first two tokens overlap
    if 'name_a' in vars and 'name_b' in vars:
      tokenized_name_a, tokenized_name_b = tokenizer.encode(vars['name_a'])[1:], tokenizer.encode(vars['name_b'])[1:]
      diff_2tok = tokenized_name_a[:2] != tokenized_name_b[:2]


    # if extract_prediction returns '<EMPTY>', skip
    fail_to_extract = False
    if extract_prediction(out) == '<EMPTY>':
      fail_to_extract = True
    # if the output DOES NOT contain any of the names, skip
    names = self.vars['name_a'] + self.vars['name_b']
    fail_to_find_name = True
    for name in names:
      if name in out:
        fail_to_find_name = False
        break
    if fail_to_extract or fail_to_find_name:
      print(f'out: {out}')
    return fail_to_extract or fail_to_find_name

  @staticmethod
  def get_inv_location_fn(tokenizer, max_input_length, inv_key):

    def get_inv_location_from_prompt(out):
      tokenized_prompt = tokenizer.batch_decode(tokenizer(out).input_ids)

      # "Story: {name_a} and {name_b} were working at the {place}. {name_c} decided to give a {object} to ...\n\nQuestion: Which of the two characters in the story likely receive the {object} in the end?\n\nAnswer:"
      # for i, x in enumerate(tokenized_prompt):
      #   print(f'{i}={len(tokenized_prompt)}-{len(tokenized_prompt)-i}',x)
      stop_sign_index = tokenized_prompt.index('.')
      decided_index = tokenized_prompt.index(' decided')
      if inv_key == 'name_c_last_tok':
          return decided_index-1
      elif inv_key == 'name_c_first_tok':
          return stop_sign_index+1
      else:
          raise ValueError(f"Invalid inv_key: {inv_key}")
    
    return get_inv_location_from_prompt