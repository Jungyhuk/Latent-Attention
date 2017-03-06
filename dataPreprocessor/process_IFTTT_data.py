import argparse
import csv
import cPickle as pickle
import collections
from HTMLParser import HTMLParser
import operator
import re
import json

import sklearn.preprocessing

# Taken from
# https://github.com/tensorflow/tensorflow/blob/16254e75e2fe4bb0f879b45fbad0c4b62c028011/tensorflow/models/rnn/translate/data_utils.py#L43
_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")


def basic_tokenizer(split_sentence):
  """Very basic tokenizer: split the sentence into a list of tokens."""
  words = []
  for space_separated_fragment in split_sentence:
    words.extend(re.split(_WORD_SPLIT, space_separated_fragment))
  return [w for w in words if w]

def read_msr_recipes_from_log(lines_iter):
  # Example entry:
  # https://ifttt.com/recipes/19-new-favorite-tweet-send-me-an-email
  #  --> 47283
  #                             ID:19
  #                          Title:New favorite tweet? Send me an email.
  #                    Description:New favorite tweet to email
  #                         Author:kev
  #                       Featured:False
  #                           Uses:37
  #                      Favorites:4
  #                           Code:(ROOT (IF) (TRIGGER (Twitter) (FUNC (New_liked_tweet_by_you) (PARAMS))) (THEN) (ACTION (Email) (FUNC (Send_me_an_email) (PARAMS (Subject ("New favorite tweet by @{{UserName}}")) (Body ({{TweetEmbedCode}}))))))

  html = HTMLParser()
  recipes_by_url = {}
  next_url = None
  while True:
    try:
      # Read the URL
      if next_url is not None:
        url = next_url
        next_url = None
      else:
        url = next(lines_iter)

      # Read the size field
      size_line = next(lines_iter)
      if size_line is None:
        break  # End of file
      elif not size_line.startswith(' -->'):
        next_url = size_line
        continue

      # Read the ID
      id_line = next(lines_iter)
      try:
        recipe_id = int(re.search(r'ID:(\d+)', id_line).group(1))
      except:
        # Seems to be a URL instead?
        next_url = id_line
        continue

      title_line = next(lines_iter)
      recipe_title = re.search(r'Title:(.*)', title_line).group(1)
      recipe_title = basic_tokenizer(html.unescape(recipe_title).lower().split(
      ))

      next(lines_iter)  # Description
      next(lines_iter)  # Author
      next(lines_iter)  # Featured
      next(lines_iter)  # Uses
      next(lines_iter)  # Favorites

      code_line = next(lines_iter)
      inside_parens = r'"[^"]+"|[^)]+'
      trigger = re.search(
        r'\(TRIGGER \(({p})\) \(FUNC \(({p})\) \(PARAMS (.*)\)\)\) \(THEN\)'.format(p=inside_parens),
          code_line)
      if trigger == None:
        trigger_channel, trigger_function = re.search(
          r'\(TRIGGER \(({p})\) \(FUNC \(({p})\)'.format(p=inside_parens),
          code_line).groups()
        trigger_param = []
      else:
        trigger_channel, trigger_function, trigger_param = trigger.groups()
        p = re.compile('\(([^\(\)]*) \(([^\(\)]*)\)\)')
        trigger_param = p.findall(trigger_param)
      action = re.search(
        r'\(ACTION \(({p})\) \(FUNC \(({p})\) \(PARAMS (.*)'.format(p=inside_parens),
          code_line)
      if action == None:
        action_channel, action_function = re.search(
          r'\(ACTION \(({p})\) \(FUNC \(({p})\)'.format(p=inside_parens),
          code_line).groups()
        action_param = []
      else:
        action_channel, action_function, action_param = action.groups()
        p = re.compile('\(([^\(\)]*) \(([^\(\)]*)\)\)')
        action_param = p.findall(action_param)
      url = url.strip()
      recipes_by_url[url] = {
          'url': url,
          'id': recipe_id,
          'recipe': recipe_title,
          'trigger_chan': trigger_channel,
          'trigger_func': trigger_channel + '.' + trigger_function,
          'trigger_func_pure': trigger_function,
          'trigger_param': trigger_param,
          'action_chan': action_channel,
          'action_func': action_channel + '.' + action_function,
          'action_func_pure': action_function,
          'action_param': action_param
      }
    except StopIteration:
      break

  return recipes_by_url


def parse_msr(log_file, extra_train, train_urls, dev_urls, test_urls,
              test_turk):
  lines_iter = iter(log_file)
  train_urls = [line.strip() for line in train_urls]
  dev_urls = [line.strip() for line in dev_urls]
  test_urls = [line.strip() for line in test_urls]

  recipes_by_url = read_msr_recipes_from_log(lines_iter)
  if extra_train:
    extra_train_recipes = read_msr_recipes_from_log(iter(extra_train))
  else:
    extra_train_recipes = None

  data = {}
  for section_name, section_urls in zip(
      ['train', 'dev', 'test'], [train_urls, dev_urls, test_urls]):
    data[section_name] = {}
    for m in ['trigger', 'action']:
      data[section_name][m] = []
      for url in section_urls:
        r = recipes_by_url.get(url, None)
        if r is None:
          continue
        data[section_name][m].append({'url': r['url'],
                                      'id': r['id'],
                                      'recipe': r['recipe'],
                                      'chan': r[m + '_chan'],
                                      'func': r[m + '_func'],
                                      'param': r[m + '_param']})
      if section_name == 'train' and extra_train_recipes:
        for r in extra_train_recipes.itervalues():
          data[section_name][m].append({'url': r['url'],
                                        'id': r['id'],
                                        'recipe': r['recipe'],
                                        'chan': r[m + '_chan'],
                                        'func': r[m + '_func'],
                                        'param': r[m + '_param']})

  if test_turk:
    reader = csv.DictReader(test_turk, delimiter='\t')
    data_by_url = collections.defaultdict(list)
    for row in reader:
      data_by_url[row['URL']].append(row)

    tagged_urls = collections.defaultdict(set)
    # omitting descriptions marked as non-English by a majority of the
    # crowdsourced workers
    english_urls = set()
    # omitting descriptions marked as either non-English or unintelligible by
    # the crowd
    intelligible_urls = set()
    # only recipes where at least three of five workers agreed with the gold
    # standard
    gold_urls = set()

    for url, rows in data_by_url.iteritems():
      non_english = 0
      unintelligible = 0
      gold = 0
      for row in rows:
        descs = (row['Trigger channel'], row['Trigger'], row['Action channel'],
                 row['Action'])

        if 'nonenglish' in descs:
          non_english += 1
          unintelligible += 1
          continue
        if 'unintelligible' in descs:
          unintelligible += 1
          continue

        descs = tuple(desc.replace(' ', '_') for desc in descs)

        gold_recipe = recipes_by_url.get(url, None)
        if gold_recipe is not None:
          if descs == (gold_recipe['trigger_chan'], gold_recipe['trigger_func_pure'],
                       gold_recipe['action_chan'], gold_recipe['action_func_pure']):
            gold += 1

      threshold = float(len(rows)) / 2
      if non_english < threshold:
        tagged_urls['english'].add(url)
      if unintelligible < threshold:
        tagged_urls['intelligible'].add(url)
      if gold > threshold:
        tagged_urls['gold'].add(url)
  else:
    tagged_urls = {}

  return data, tagged_urls


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--output', required=True)
  parser.add_argument('--log', help='Path to log from MSR crawler')
  parser.add_argument('--extra-train')
  parser.add_argument('--train-urls',
                          help='List of URLs to use for training.')
  parser.add_argument('--dev-urls',
                          help='List of URLs to use for development.')
  parser.add_argument('--test-urls',
                          help='List of URLs to use for testing.')
  parser.add_argument('--test-turk', help='MTurk results on test set.')

  args = parser.parse_args()

  tagged_urls = {}
  with open(args.log) as log, open(args.train_urls) as train_urls, \
      open(args.dev_urls) as dev_urls, open(args.test_urls) as test_urls, \
      open(args.test_turk) as test_turk:
    extra_train = (open(args.extra_train) if args.extra_train is not None
                     else None)
    data, tagged_urls = parse_msr(log, extra_train, train_urls, dev_urls,
                                    test_urls, test_turk)

  # Assign IDs to each word and label in the training data
  vocab = collections.Counter()
  for item in data['train']['action']:
    for word in item['recipe']:
      vocab[word] += 1
  words_dec_freq = sorted(vocab.iteritems(),
                          key=operator.itemgetter(1),
                          reverse=True)
  word_ids = {k: i for i, (k, count) in enumerate(words_dec_freq)}

  fout = open('IFTTT_words.json','w')

  word_list = []
  for item in words_dec_freq:
    new_item = {}
    new_item["word"] = item[0]
    new_item["freq"] = item[1]
    word_list.append(new_item)
  fout.write(json.dumps(word_list, indent = 1))

  # Make {train,test}_{trigger,action}_{channels,functions}.
  all_labels = collections.defaultdict()
  labelers = {}
  for m in ['trigger', 'action']:
    for n in ['chans', 'funcs']:
      label_type = m + '_' + n
      labels = {}

      for section in ['train', 'test', 'dev']:
        if section not in data: continue
        labels[section] = map(operator.itemgetter(n[:-1]), data[section][m])

      labeler = sklearn.preprocessing.LabelEncoder()
      labeler.fit(labels['train'] + labels['test'])

      for section in ['train', 'test', 'dev']:
        if section not in data: continue
        labels[section] = labeler.transform(labels[section])

      all_labels[label_type] = labels
      labelers[label_type] = labeler

  label_types = ('trigger_chans', 'trigger_funcs', 'action_chans',
                 'action_funcs')

  train_params = {}
  for section in ['train', 'test', 'dev']:
    for m in ['trigger','action']:
      for item in data[section][m]:
        params = item['param']
        if not (m+'/'+item['func'] in train_params):
          train_params[m+'/'+item['func']] = {}
        for param_name_pure, param_value in params:
          if not (param_name_pure in train_params[m+'/'+item['func']]):
            train_params[m+'/'+item['func']][param_name_pure] = {}
            train_params[m+'/'+item['func']][param_name_pure]["<NULL>"] = 0
          if not (param_value in train_params[m+'/'+item['func']][param_name_pure]):
            train_params[m+'/'+item['func']][param_name_pure][param_value] = 0
          if section == 'train':
            train_params[m+'/'+item['func']][param_name_pure][param_value] += 1

  for m in ['trigger','action']:
    for item in data['train'][m]:
        params = item['param']
        for param_name in train_params[m+'/'+item['func']]:
          param_value = '<NULL>'
          for param in params:
            if param[0] == param_name:
              param_value = param[1]
              break
          if param_value == '<NULL>':
            train_params[m+'/'+item['func']][param_name][param_value] += 1

  predicted_params = {}
  param_num = {}
  for func_name in train_params:
    if not (func_name in predicted_params):
      predicted_params[func_name] = {}
      param_num[func_name] = 0
    for param_name in train_params[func_name]: 
      param_values = train_params[func_name][param_name]
      predicted_value = sorted(param_values.iteritems(), key=operator.itemgetter(1), reverse=True)[:1]
      if predicted_value[0][0] != '' and (train_params[func_name][param_name]['<NULL>'] == train_params[func_name][param_name][predicted_value[0][0]] or train_params[func_name][param_name][predicted_value[0][0]] <= 1):
        predicted_params[func_name][param_name] = '<NULL>'
      else:
        predicted_params[func_name][param_name] = predicted_value[0][0]
      if predicted_params[func_name][param_name] != '<NULL>':
        param_num[func_name] += 1

  outputs = {}
  for section in ['train', 'test', 'dev']:
    if section not in data: continue
    output = []
    for i in xrange(len(data[section]['action'])):
      words = data[section]['action'][i]['recipe']
      item = {'ids': [word_ids.get(word, len(word_ids)) for word in words],
              'labels': [all_labels[t][section][i] for t in label_types],
              'label_names': [labelers[t].classes_[all_labels[t][section][i]]
                              for t in label_types],
              'words': words, 
              'params': [data[section][t][i]['param'] for t in ['trigger','action']]}
      correct_trigger_param = 0
      correct_action_param = 0
      semi_correct_trigger_param = 0
      semi_correct_action_param = 0
      for param_name in train_params['trigger'+'/'+item['label_names'][1]]:
        truth = '<NULL>'
        for param in data[section]['trigger'][i]['param']:
          if param[0] == param_name:
            truth = param[1]
            break 
        if predicted_params['trigger'+'/'+item['label_names'][1]][param_name] == truth and truth != '<NULL>':
          correct_trigger_param += 1
        else:
          if truth != '<NULL>' and predicted_params['trigger'+'/'+item['label_names'][1]][param_name] != '<NULL>':
            semi_correct_trigger_param += 1
      for param_name in train_params['action'+'/'+item['label_names'][3]]:
        truth = '<NULL>'
        for param in data[section]['action'][i]['param']:
          if param[0] == param_name:
            truth = param[1]
            break
        if predicted_params['action'+'/'+item['label_names'][3]][param_name] == truth  and truth != '<NULL>':
          correct_action_param += 1
        else:
          if truth != '<NULL>' and predicted_params['action'+'/'+item['label_names'][3]][param_name] != '<NULL>':
            semi_correct_action_param += 1 
      item['correct_trigger_param'] = correct_trigger_param
      item['correct_action_param'] = correct_action_param
      item['semi_correct_trigger_param'] = semi_correct_trigger_param
      item['semi_correct_action_param'] = semi_correct_action_param
      if 'url' in data[section]['action'][i]:
        item['url'] = data[section]['action'][i]['url']
        tags = []
        for tag_name, tag_set in tagged_urls.iteritems():
          if item['url'] in tag_set:
            tags.append(tag_name)
        if tags:
          item['tags'] = tags

      output.append(item)
    outputs[section] = output
  outputs['label_types'] = label_types
  outputs['labelers'] = labelers
  outputs['word_ids'] = word_ids
  outputs['num_labels'] = [len(labelers[t].classes_) for t in label_types]
  outputs['train_params'] = train_params
  outputs['predicted_params'] = predicted_params
  outputs['param_num'] = param_num

  pickle.dump(outputs, open(args.output, 'w'), pickle.HIGHEST_PROTOCOL)
