import argparse
import cPickle as pickle
import collections
import numpy as np

label_names = ['trigger_chans', 'trigger_funcs', 'action_chans',
               'action_funcs', 'tc+ac', 'tc+tf+ac+af']

def compute_accuracy_from_probs(probs, labels):
  pred_labels = np.vstack([np.argmax(arr, axis=1) for arr in probs]).T
  correct = pred_labels == labels
  correct = np.concatenate(
      (correct, np.all(correct[:, [0, 2]], axis=1)[:, np.newaxis],
       np.all(correct[:, [1, 3]], axis=1)[:, np.newaxis]),
      axis=1)
  for i, name in enumerate(label_names):
    print name, np.sum(correct[:, i]) / float(len(correct[:, i]))
  print

def compute_F1(args, probs_trigger, probs_action, labels, raw_data):
  precision = 0
  recall = 0
  tot_precision = 0
  tot_recall = 0
  input_data = pickle.load(open(args.data))
  labelers = input_data['labelers']
  train_params = input_data['train_params']
  param_num = input_data['param_num']
  for i in xrange(len(labels)):
    pred_trigger = np.argmax(probs_trigger[i])
    pred_trigger_name = labelers['trigger_funcs'].classes_[pred_trigger]
    pred_trigger_channel= pred_trigger_name[:pred_trigger_name.find('.')]
    trigger_name = labelers['trigger_funcs'].classes_[labels[i][1]]
    trigger_channel = trigger_name[:trigger_name.find('.')]

    pred_action = np.argmax(probs_action[i])
    pred_action_name = labelers['action_funcs'].classes_[pred_action]
    pred_action_channel= pred_action_name[:pred_action_name.find('.')]
    action_name = labelers['action_funcs'].classes_[labels[i][3]]
    action_channel = action_name[:action_name.find('.')]

    tot_precision += 1
    tot_recall += 1
    precision += 1
    recall += 1

    tot_precision += 1
    tot_recall += 1
    if pred_trigger_channel == trigger_channel:
      precision += 1
      recall += 1

    tot_precision += 1
    tot_recall += 1
    if pred_trigger_name == trigger_name:
      precision += 1
      recall += 1
    tot_precision += param_num['trigger'+'/'+pred_trigger_name] * 2
    tot_recall += len(raw_data[i]['params'][0]) * 2
    if pred_trigger_name == trigger_name:
      precision += raw_data[i]['correct_trigger_param'] * 2 + raw_data[i]['semi_correct_trigger_param']
      recall += raw_data[i]['correct_trigger_param'] * 2 + raw_data[i]['semi_correct_trigger_param']
          
    tot_precision += 1
    tot_recall += 1
    precision += 1
    recall += 1

    tot_precision += 1
    tot_recall += 1
    if pred_action_channel == action_channel:
      precision += 1
      recall += 1
    tot_precision += 1
    tot_recall += 1
    if pred_action_name == action_name:
      precision += 1
      recall += 1
    tot_precision += param_num['action'+'/'+pred_action_name] * 2
    tot_recall += len(raw_data[i]['params'][1]) * 2
    if pred_action_name == action_name:
      precision += raw_data[i]['correct_action_param'] * 2 + raw_data[i]['semi_correct_action_param']
      recall += raw_data[i]['correct_action_param'] * 2 + raw_data[i]['semi_correct_action_param']

  precision = precision * 1.0 / tot_precision
  recall = recall * 1.0 / tot_recall

  print 'F1 score: ', 2 * precision * recall / (precision + recall)
    

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--res', nargs='+')
  parser.add_argument('--data')
  parser.add_argument('--F1', action='store_true')
  args = parser.parse_args()

  # section -> label type -> list?
  all_probs = collections.defaultdict(lambda: [[] for i in xrange(4)])

  input_data = pickle.load(open(args.data))
  test_data = input_data['test']

  for path in args.res:
    data = pickle.load(open(path))
    print path
    print '================='

    for sect in data['probs']:
      probs = [np.array(ls) for ls in data['probs'][sect]]
      for label_type, prob in enumerate(probs):
        all_probs[sect][label_type].append(prob)

      labels = np.array(data['labels'][sect]).T
      print sect
      print '--------------------'
      compute_accuracy_from_probs(probs, labels)
      if args.F1:
        raw_data = []
        for item in test_data:
          if sect == 'test' or (('tags' in item) and (sect[sect.find('-')+1:] in item['tags'])):
            raw_data.append(item)
        compute_F1(args, probs[1], probs[3], labels, raw_data)

  print 'averaged'
  print '=========================='

#  import IPython
#  IPython.embed()
  for sect in all_probs:
    probs = [np.mean(arrs, axis=0) for arrs in all_probs[sect]]

    labels = np.array(data['labels'][sect]).T

    print sect
    print '--------------------'
    compute_accuracy_from_probs(probs, labels)
    if args.F1:
      raw_data = []
      for item in test_data:
        if sect == 'test' or (('tags' in item) and (sect[sect.find('-')+1:] in item['tags'])):
          raw_data.append(item)
      compute_F1(args, probs[1], probs[3], labels, raw_data)

