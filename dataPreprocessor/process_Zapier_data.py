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

def parse_recipes(input_data):
  # Example entry:
  #"title": "Tweet New RSS Feed Item", 
  #"description": "Automatically tweet new RSS feed items.", 
  #"action_channel": "TwitterV2API", 
  #"event_channel": "RSSAPI", 
  #"action": "tweet",
  #"event": "new_feed", 
  #"rule": "{u'message': u'{{title}}: {{link}}'}"
  recipes = []
  raw_data = json.load(input_data)
  for item in raw_data:
      if item["event_channel"] == None:
        item["event_channel"]="None"
      if item["action_channel"] == None:
        item["action_channel"]="None"
      if item["event"] == None:
        item["event"]="None"
      if item["action"] == None:
        item["action"]="None"
      recipes.append({
          'recipe': item["title"],
          'trigger_chan': item["event_channel"],
          'trigger_func': item["event_channel"] + '.' + item["event"],
          'trigger_func_pure': item["event"],
          'action_chan': item["action_channel"],
          'action_func': item["action_channel"] + '.' + item["action"],
          'action_func_pure': item["action"],
          'rule': item["rule"]
      })
  return recipes


def parse_Zapier(input_train, input_dev, input_test):
  training_recipes = parse_recipes(input_train)
  dev_recipes = parse_recipes(input_dev)
  test_recipes = parse_recipes(input_test)
  data = {}
  for section_name, section_data in zip(
      ['train', 'dev','test'], [training_recipes, dev_recipes, test_recipes]):
    data[section_name] = {}
    for m in ['trigger', 'action']:
      data[section_name][m] = []
      for item in section_data:
        data[section_name][m].append({'recipe': basic_tokenizer(item['recipe'].lower().split()),
                                      'chan': item[m + '_chan'],
                                      'func': item[m + '_func'],
                                     })
  return data


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--output', required=True)
  parser.add_argument('--input-train', help='Path to Zapier training data(json file)')
  parser.add_argument('--input-dev', help='Path to Zapier dev data(json file)')
  parser.add_argument('--input-test', help='Path to Zapier test data(json file)')
  parser.add_argument('--word-list', help='Path to word list(json file)')
  args = parser.parse_args()

  with open(args.input_train) as input_train, open(args.input_dev) as input_dev, open(args.input_test) as input_test:
      data = parse_Zapier(input_train, input_dev, input_test)

  # Assign IDs to each word and label in the training data
  vocab = collections.Counter()
  for item in data['train']['action']:
    for word in item['recipe']:
      vocab[word] += 1
  words_dec_freq = sorted(vocab.iteritems(),
                          key=operator.itemgetter(1),
                          reverse=True)

  fout = open('Zapier_words.json','w')

  word_list = []
  for item in words_dec_freq:
    new_item = {}
    new_item["word"] = item[0]
    new_item["freq"] = item[1]
    word_list.append(new_item)
  fout.write(json.dumps(word_list, indent = 1))

  if args.word_list:
    with open(args.word_list) as f:
      word_list = json.load(f)
    word_ids = {}
    for item in word_list:
      word_ids[item["word"]] = len(word_ids)
  else:
    word_ids = {k: i for i, (k, count) in enumerate(words_dec_freq)}

  # Make {train,test}_{trigger,action}_{channels,functions}.
  all_labels = collections.defaultdict()
  labelers = {}
  for m in ['trigger', 'action']:
    for n in ['chans', 'funcs']:
      label_type = m + '_' + n
      labels = {}

      for section in ['train', 'dev', 'test']:
        if section not in data: continue
        labels[section] = map(operator.itemgetter(n[:-1]), data[section][m])

      labeler = sklearn.preprocessing.LabelEncoder()
      labeler.fit(labels['train'] + labels['test'])

      for section in ['train', 'dev', 'test']:
        if section not in data: continue
        labels[section] = labeler.transform(labels[section])

      all_labels[label_type] = labels
      labelers[label_type] = labeler

  label_types = ('trigger_chans', 'trigger_funcs', 'action_chans',
                 'action_funcs')

  outputs = {}
  for section in ['train', 'dev', 'test']:
    output = []
    for i in xrange(len(data[section]['action'])):
      words = data[section]['action'][i]['recipe']
      item = {'ids': [word_ids.get(word, len(word_ids)) for word in words],
              'labels': [all_labels[t][section][i] for t in label_types],
              'label_names': [labelers[t].classes_[all_labels[t][section][i]]
                              for t in label_types],
              'words': words}
      output.append(item)
      #output.append({'words': words, 'ids': ids, 'labels': labels})
    outputs[section] = output
  outputs['label_types'] = label_types
  outputs['labelers'] = labelers
  outputs['word_ids'] = word_ids
  outputs['num_labels'] = [len(labelers[t].classes_) for t in label_types]

  pickle.dump(outputs, open(args.output, 'w'), pickle.HIGHEST_PROTOCOL)
