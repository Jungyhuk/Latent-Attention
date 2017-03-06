import argparse, bisect
import collections
import cPickle as pickle
import hashlib, imp
import json, random
import os, shutil, sys
import tempfile
import time

import numpy as np
import tensorflow as tf
import tf_utils
import _jsonnet

from model import *

class IftttTrain(tf_utils.TFMainLoop):
  def __init__(self, args, config):
    self.config = config
    self.optim_config = config['optim']
    self.model_config = config['model']
    self.eval_config = config['eval']
    self.arch_config = config.get('architecture', None)
    self.root_logdir = args.logdir
    self.output_file = args.output

    self.data = pickle.load(open(args.dataset))
    if 'label_types' in self.data:
      global label_names
      label_names = self.data['label_types']
    self.max_desc_length = max(
      len(item['ids']) for item in \
      self.data['train'] + self.data['dev'] + self.data['test'])
    self.memory_size = self.model_config.get('memory_size', self.max_desc_length)

    for section in ('train', 'test', 'dev'):
      for i in xrange(len(self.data[section])):
        if len(self.data[section][i]['ids']) > self.memory_size:
          d = self.data[section][i]['ids']
          d = d[:(self.memory_size + 1) // 2] + d[len(d) - self.memory_size//2:]
          self.data[section][i]['ids'] = d

    self.max_desc_length = max(
      len(item['ids']) for item in \
      self.data['train'] + self.data['dev'] + self.data['test'])

    print('max length:',self.max_desc_length)

    buckets = tf_utils.create_buckets(
        [len(item['ids']) for item in self.data['train']],
        self.optim_config['batch_size'] * 5000)
    print 'Buckets:', buckets

    self.bucketed_train = [[] for i in range(len(buckets))]
    for item in self.data['train']:
      size = len(item['ids'])
      self.bucketed_train[bisect.bisect_left(buckets, size)].append(item)
    bucketed_train_lens = np.array(
      [len(bucket) for bucket in self.bucketed_train],
      dtype=float)
    self.bucket_dist = bucketed_train_lens / np.sum(bucketed_train_lens)
    self.initializer = getattr(tf, self.optim_config['init']['name'])(
      -self.optim_config['init']['scale'],
      self.optim_config['init']['scale'])

    max_word_id = int(self.arch_config['max_word_id'] if self.arch_config else
                      config['max_word_id'])
    if max_word_id > 0:
      self.vocab_size = max_word_id
    else:
      self.vocab_size = len(self.data['word_ids'])

    self.last_test_time = 0
    if self.eval_config['unit'] == 'epochs':
      num_steps_per_epoch = (len(self.data['train']) /
                             self.optim_config['batch_size'])
      self.num_steps_per_eval = int(num_steps_per_epoch *
                                    self.eval_config['freq'])
      print 'Testing every {} steps.'.format(self.num_steps_per_eval)
    elif self.eval_config['unit'] == 'steps':
      self.num_steps_per_eval = self.eval_config['freq']

    # Eval measures
    self.label_types = list(self.arch_config['label_types'] if self.arch_config
                            else self.config['label_types'])
    self.label_type_names = list(np.array(label_names)[self.label_types])

    ## XXX
    if self.label_types == [0, 1, 2, 3]:
      self.label_type_names.append('tc+ac')
      self.label_type_names.append('tc+tf+ac+af')

    # Tracking accuracies
    self.all_accuracies_by_section = collections.defaultdict(dict)
    self.best_accuracy = np.zeros(len(self.label_type_names))
    self.best_iters = np.zeros(len(self.label_type_names))
    self.num_tests_below_best_accuracy = 0
    self.stop_training = False

  def read_batch(self, current_bucket, current_bucket_start):
    bucket = self.bucketed_train[current_bucket]
    batch = bucket[current_bucket_start:current_bucket_start + self.optim_config['batch_size']]
    batch = batch + [random.choice(bucket)
             for _ in range(self.optim_config['batch_size'] - len(batch))]
    # batch x time
    ids = tf_utils.make_array(
        [item['ids'] for item in batch],
        length=self.memory_size)
    # batch
    ids_lengths = np.array([len(item['ids']) for item in batch])
    # batch x label types
    labels = tf_utils.make_array([item['labels'] for item in batch
                                  ])[:, self.label_types]
    # batch x time x label types
    expanded_labels = np.zeros((labels.shape[0], ids.shape[1], labels.shape[1]
                                ))
    expanded_labels[:, ids_lengths - 1] = labels

    return ids, ids_lengths, labels, expanded_labels

  def train_feed_dict(self, m, data):
    ids, ids_lengths, labels, expanded_labels = data

    feed_dict = tf_utils.filter_none_keys({m.ids: ids,
                                           m.ids_lengths: ids_lengths,
                                           m.labels: labels,
                                           m.expanded_labels: expanded_labels})
    return feed_dict

  def create_model(self, is_training):
    if self.arch_config:
      label_types = list(self.arch_config['label_types'])
    else:
      label_types = list(self.label_types)
    return IFTTTModel(self.optim_config,
                      self.model_config,
                      self.arch_config,
                      np.array(self.data['num_labels'])[label_types],
                      self.label_types,
                      self.vocab_size,
                      self.memory_size,
                      is_training)

  def should_test_model(self, global_step):
    if self.eval_config['unit'] in ('epochs', 'steps'):
      return global_step % self.num_steps_per_eval == 1
    elif self.eval_config['unit'] == 'seconds':
      if time.time() - self.last_test_time > 60:
        self.last_test_time = time.time()
        return True
      return False
    else:
      raise ValueError('invalid eval/unit: ' + self.eval_config['unit'])

  def eval_summary_writers(self, logdir):
    return keydefaultdict(
        lambda section: tf.train.SummaryWriter(os.path.join(logdir, section)))

  def should_stop_training(self):
    return self.stop_training

  def generate_test_summaries(self, session, mtest, global_step):
    num_correct = collections.defaultdict(
        lambda: np.zeros(len(self.label_type_names), dtype=np.float))
    total = collections.defaultdict(int)
    for section in ('train', 'test', 'dev'):
      if section not in self.data:
        continue
      batch_size = self.optim_config['batch_size']
      for i in xrange(0, len(self.data[section]),
                      batch_size):
        batch = self.data[section][i:i + batch_size]
        nc, size, preds = self.eval_batch(session, mtest, batch, batch_size)

        for i, row in enumerate(batch):
          total[section] += 1
          num_correct[section] += nc[i]
          for tag in row.get('tags', []):
            tagged_section = '{}-{}'.format(section, tag)
            total[tagged_section] += 1
            num_correct[tagged_section] += nc[i]

    for section, correct in num_correct.iteritems():
      self.all_accuracies_by_section[section][global_step] = correct / total[
          section]

    all_results = {section: [
        tf.Summary.Value(tag='test_acc/{}'.format(label_name),
                         simple_value=float(self.all_accuracies_by_section[
                             section][global_step][i]))
        for i, label_name in enumerate(self.label_type_names)
    ]
                   for section in num_correct.keys()}
    train_results = all_results['train']
    del all_results['train']
    other_results = all_results

    if 'dev' in self.data:
      candidate_best_acc = np.array(num_correct['dev']) / total['dev']
    else:
      candidate_best_acc = np.array(num_correct['test']) / total['test']

    if np.all(candidate_best_acc < self.best_accuracy):
      self.num_tests_below_best_accuracy += 1
      print 'Failed to exceed best accuracy: {}'.format(
          self.num_tests_below_best_accuracy)
    else:
      improved_categories = candidate_best_acc >= self.best_accuracy
      if improved_categories[1]:
	probs_by_section = collections.defaultdict(
            lambda: [[] for i in xrange(4)])
        labels_by_section = collections.defaultdict(
            lambda: [[] for i in xrange(4)])

        batch_size = self.optim_config['batch_size']
        for i in xrange(0, len(self.data['test']), batch_size):
          batch = self.data['test'][i:i + batch_size]
          all_probs = self.eval_batch(session,
                                      mtest,
                                      batch,
                                      batch_size,
                                      get_probs=True)
          assert len(all_probs) == 4

          for row_index, row in enumerate(batch):
            for tag in row.get('tags', []):
              tagged_section = '{}-{}'.format('test', tag)
              for prob_matrix, container in zip(
                  all_probs, probs_by_section[tagged_section]):
                container.append(prob_matrix[row_index])

              for label, container in zip(row['labels'],
                                          labels_by_section[tagged_section]):
                container.append(label)
          sys.stdout.write(".")
          sys.stdout.flush()
        print

        pickle.dump(
          {'probs': dict(probs_by_section),
           'labels': dict(labels_by_section)}, open(
               self.output_file+'.pkl', 'w'),
          pickle.HIGHEST_PROTOCOL)
      self.best_accuracy[improved_categories] = candidate_best_acc[
          improved_categories]
      self.best_iters[improved_categories] = global_step

      if np.all(improved_categories[self.eval_config['relevant_labels']]):
        self.num_tests_below_best_accuracy = 0
      else:
        self.num_tests_below_best_accuracy += 1
      print 'Num tests below best accuracy: {}'.format(
          self.num_tests_below_best_accuracy)

    print 'Keys: {}'.format(self.label_type_names)
    print 'This time: {}'.format(candidate_best_acc)
    print 'Current best: {}'.format(self.best_accuracy)
    print 'Best iters: {}'.format(self.best_iters)
    for section in num_correct:
      print '{} best: {}'.format(
          section, np.array([self.all_accuracies_by_section[section][
              self.best_iters[i]][i] for i in xrange(len(self.best_iters))]))

    # Write stats
    stats = {
        'keys': list(self.label_type_names),
        'best_accuracy': list(self.best_accuracy),
        'best_iters': list(self.best_iters),
        'num_tests_below_best': self.num_tests_below_best_accuracy,
    }
    with open(os.path.join(self.root_logdir, 'stats.json.new'), 'w') as f:
      json.dump(stats, f)
    os.rename(
        os.path.join(self.root_logdir, 'stats.json.new'),
        os.path.join(self.root_logdir, 'stats.json'))

    if (self.num_tests_below_best_accuracy >=
        self.eval_config['max_unsuccessful_trials']):
      self.stop_training = True

    return train_results, other_results

  def eval_batch(self, session, mtest, batch, batch_size, get_probs=False):
    batch_size_original = len(batch)
    ids = tf_utils.make_array(
      [item['ids'] for item in batch] + \
      [batch[0]['ids'] for _ in xrange(batch_size - len(batch))],
      length=self.memory_size)
    ids_lengths = np.array(
      [len(item['ids']) for item in batch] + \
      [len(batch[0]['ids']) for _ in xrange(batch_size - len(batch))])
    labels = tf_utils.make_array(
      [item['labels'] for item in batch] + \
      [batch[0]['labels'] for _ in xrange(batch_size - len(batch))])
    labels = labels[:, list(self.label_types)]
    if get_probs:
      all_probs = session.run(mtest.all_probs, tf_utils.filter_none_keys({
          mtest.ids: ids,
          mtest.ids_lengths: ids_lengths,
          mtest.labels: labels,
      }))
      return all_probs[:batch_size_original]

    (preds, ) = session.run([mtest.preds], tf_utils.filter_none_keys({
        mtest.ids: ids,
        mtest.ids_lengths: ids_lengths,
        mtest.labels: labels,
    }))
    preds = preds[:batch_size_original]
    labels = labels[:batch_size_original]
    correct = (preds == labels)

    if self.label_types == [0, 1, 2, 3]:
      correct = np.concatenate(
          (correct, np.all(correct[:, [0, 2]], axis=1)[:, np.newaxis],
           np.all(correct[:, [1, 3]], axis=1)[:, np.newaxis]),
          axis=1)

    return correct, len(preds), preds

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset', required=True)
  parser.add_argument('--load-model')
  parser.add_argument('--config')
  parser.add_argument('--number-logdir', action='store_true')
  parser.add_argument('--test-logdir', action='store_true')
  parser.add_argument('--logdir')
  parser.add_argument('--clear', action='store_true')
  parser.add_argument('--log-device-placement', action='store_true')
  parser.add_argument('--output', required=True)
  args = parser.parse_args()

  if args.logdir is None:
    args.logdir = tempfile.mkdtemp(prefix='ifttt_')
    print args.logdir
  if args.clear and not (args.number_logdir or args.test_logdir):
    try:
      shutil.rmtree(args.logdir)
    except OSError:
      pass

  if args.test_logdir:
    config = json.load(open(os.path.join(args.logdir, 'config.json')))
    stats = json.load(open(os.path.join(args.logdir, 'stats.json')))

    ifttt_train = IftttTrain(args, config)

    with tf.variable_scope('model', reuse=None, initializer=None):
      m = ifttt_train.create_model(is_training=False)

    for best_iter, name in zip(stats['best_iters'], stats['keys']):
      print name
      saver = tf.train.Saver(max_to_keep=0)
      with tf.Session() as sess:
        saver.restore(sess,
                      os.path.join(args.logdir,
                                   'model.ckpt-{}'.format(int(best_iter))))
        # section type -> label type -> rows
        probs_by_section = collections.defaultdict(
            lambda: [[] for i in xrange(4)])
        labels_by_section = collections.defaultdict(
            lambda: [[] for i in xrange(4)])

        batch_size = ifttt_train.optim_config['batch_size']
        for i in xrange(0, len(ifttt_train.data['test']), batch_size):
          batch = ifttt_train.data['test'][i:i + batch_size]

          all_probs = ifttt_train.eval_batch(sess, m, batch, batch_size,
                                             get_probs=True)
          assert len(all_probs) == 4

          for row_index, row in enumerate(batch):
            for tag in row.get('tags', []):
              tagged_section = '{}-{}'.format('test', tag)
              for prob_matrix, container in zip(
                  all_probs, probs_by_section[tagged_section]):
                container.append(prob_matrix[row_index])

              for label, container in zip(row['labels'],
                                          labels_by_section[tagged_section]):
                container.append(label)
          sys.stdout.write(".")
          sys.stdout.flush()
        print

      pickle.dump(
          {'probs': dict(probs_by_section),
           'labels': dict(labels_by_section)}, open(
               os.path.join(args.logdir, 'probs-{}.pkl'.format(name)), 'w'),
          pickle.HIGHEST_PROTOCOL)

  if args.config:
    pretty_config_str = _jsonnet.evaluate_file(args.config)
    print pretty_config_str
    tf_utils.mkdir_p(args.logdir)
    if args.number_logdir:
      sub_logdirs = os.listdir(args.logdir)
      sub_logdirs.sort()
      logdir_id = int(sub_logdirs[-1]) + 1 if sub_logdirs else 0
      args.logdir = os.path.join(args.logdir, '{:06d}'.format(logdir_id))
      os.mkdir(args.logdir)

    with open(os.path.join(args.logdir, 'config.json'), 'w') as f:
      f.write(pretty_config_str)

    config = json.loads(pretty_config_str)
    ifttt_train = IftttTrain(args, config)
    try:
      ifttt_train.run(ifttt_train.initializer, args.logdir,
                      args.log_device_placement, args.load_model)
    except KeyboardInterrupt:
      pass

    print 'Config:'
    print pretty_config_str
    print 'Label names:', ifttt_train.label_type_names
    print 'Best accuracies:', ifttt_train.best_accuracy
    print 'Best iters:', ifttt_train.best_iters
    print 'Logdir:', args.logdir

if __name__ == '__main__':
  main()
