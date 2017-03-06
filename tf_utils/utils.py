import time
import os
import errno

import numpy as np
import tensorflow as tf
import random


def tf_print(tensor, message=None, summarize=None):
  return tf.Print(tensor, [tensor], message=message, summarize=summarize)


def tf_print_shape(tensor, message=None, summarize=None):
  return tf.Print(tensor, [tf.shape(tensor)],
                  message=message,
                  summarize=summarize)


def make_array(seqs, length=None):
  '''Make a 2D NumPy array from a list of strings or a list of 1D arrays/lists.
  Shape of result is len(seqs) x length of longest sequence.'''
  if length is None:
    length = max(len(elem) for elem in seqs)
  array = np.full((len(seqs), length), 0, dtype=np.int32)

  for i, item in enumerate(seqs):
    if isinstance(item, str):
      item = np.fromstring(item, np.uint8)
    array[i, :len(item)] = item

  return array


def filter_none_keys(d):
  return {k: v for k, v in d.iteritems() if k is not None}


def mkdir_p(path):
  try:
    os.makedirs(path)
  except OSError as exc:  # Python >2.5
    if exc.errno == errno.EEXIST and os.path.isdir(path):
      pass
    else:
      raise


class EMAverager(object):
  def __init__(self, decay):
    self.average = None
    self.decay = decay

  def add(self, value):
    if self.average is None:
      self.average = value
    else:
      self.average *= self.decay
      self.average += (1 - self.decay) * value

  def get(self):
    return self.average


class TFMainLoop(object):
  def create_model(self, is_training):
    raise NotImplementedError

  def read_batch(self):
    raise NotImplementedError

  def train_feed_dict(self, model, data):
    raise NotImplementedError

  def generate_train_summaries(self, cost):
    return []

  def should_test_model(self, global_step):
    return False

  def eval_summary_writers(self, logdir):
    return {'test': tf.train.SummaryWriter(os.path.join(logdir, 'test'))}

  def should_stop_training(self):
    pass

  def generate_test_summaries(self, session, mtest, global_step):
    raise NotImplementedError

  def initialize_extra_vars(self, session, model):
    pass

  def run(self, initializer, logdir, log_device_placement=False, load_model=None):
    with tf.variable_scope('model', reuse=None, initializer=initializer):
      m = self.create_model(is_training=True)
    with tf.variable_scope('model', reuse=True, initializer=initializer):
      mvalid = self.create_model(is_training=False)
    saver = tf.train.Saver(max_to_keep=0)
    session = tf.Session()
    if load_model:
      saver.restore(session, load_model)
    else:
      session.run(tf.initialize_all_variables())
    # Disable automatic saving and deleting
    supervisor = tf.train.Supervisor(logdir=logdir,
                                     summary_op=None,
                                     save_model_secs=0,
                                     saver=saver)
    #session = supervisor.prepare_or_wait_for_session(
    #    config=tf.ConfigProto(log_device_placement=log_device_placement))      
    eval_summary_writers = self.eval_summary_writers(logdir)
    #test_summary_writer = tf.train.SummaryWriter(os.path.join(logdir, 'test'))

    self.initialize_extra_vars(session, m)

    step_elapsed_time_avg = EMAverager(0.01)
    step_elapsed_time_scaled_avg = EMAverager(0.01)
    last_print_time = 0
    current_bucket = 0
    current_bucket_start = 0
    random.shuffle(self.bucketed_train)
    while not supervisor.should_stop() and not self.should_stop_training():
      step_start_time = time.time()

      # Get a batch of data
      data_read_start_time = time.time()
      if current_bucket_start == 0:
          random.shuffle(self.bucketed_train[current_bucket])
      batch = self.read_batch(current_bucket, current_bucket_start)
      if current_bucket_start + len(batch[0]) >= len(self.bucketed_train[current_bucket]):
        current_bucket = current_bucket + 1
        if current_bucket >= len(self.bucketed_train):
          random.shuffle(self.bucketed_train)
          current_bucket = 0
        current_bucket_start = 0
      else:
        current_bucket_start += len(batch[0])
      data_read_elapsed_time = time.time() - data_read_start_time
      #      print 'Reading data took {} seconds'.format(data_read_elapsed_time)

      # Run 1 step with minibatch
      cost, _, summaries, global_step = session.run(
          [m.loss, m.train_op, m.summaries, m.global_step],
          self.train_feed_dict(m, batch))

      step_elapsed_time = time.time() - step_start_time
      # TODO replace 100 with real number
      step_elapsed_time_scaled = step_elapsed_time / 100
      step_elapsed_time_avg.add(step_elapsed_time)
      step_elapsed_time_scaled_avg.add(step_elapsed_time_scaled)

      summaries = tf.Summary.FromString(summaries)
      summaries.value.extend([
          tf.Summary.Value(tag='step_time/volatile/raw',
                           simple_value=step_elapsed_time),
          tf.Summary.Value(tag='step_time/volatile/scaled',
                           simple_value=step_elapsed_time_scaled),
          tf.Summary.Value(tag='step_time/average/raw',
                           simple_value=step_elapsed_time_avg.get()),
          tf.Summary.Value(tag='step_time/average/scaled',
                           simple_value=step_elapsed_time_scaled_avg.get()),
      ])

      supervisor.summary_writer.add_summary(summaries.SerializeToString(),
                                            global_step)

      if self.should_test_model(global_step):
        eval_start_time = time.time()

        train_summaries, eval_summaries = self.generate_test_summaries(
            session, mvalid, global_step)

        train_summary = tf.Summary(value=train_summaries + [tf.Summary.Value(
            tag='eval_time',
            simple_value=time.time() - eval_start_time), ])

        supervisor.summary_writer.add_summary(train_summary, global_step)
        supervisor.summary_writer.flush()

        for name, summary_values in eval_summaries.iteritems():
          summary = tf.Summary(value=summary_values)
          eval_summary_writers[name].add_summary(summary, global_step)
          eval_summary_writers[name].flush()

        print 'test step {}:'.format(global_step)
        print 'train: {}'.format(train_summary)
        for k, v in eval_summaries.iteritems():
          print '{}: {}'.format(k, tf.Summary(value=v))

        # Save model
        supervisor.saver.save(session,
                              supervisor.save_path,
                              global_step=global_step)
        print 'Saved model at global step {}'.format(global_step)

      now = time.time()
      if now - last_print_time > 5:
        #print 'step {}: {}'.format(global_step, summaries)
        last_print_time = now
        supervisor.summary_writer.flush()

    # Clean up
    supervisor.summary_writer.flush()
    for summary_writer in eval_summary_writers.itervalues():
      summary_writer.flush()


def create_buckets(sizes, min_bucket_size):
  '''Determine upper bounds for dividing |sizes| into buckets (contiguous
     ranges) of approximately equal size, where each bucket has at least
     |min_bucket_size|.'''

  sizes.sort()

  buckets = []
  bucket = []
  for size in sizes:
    if len(bucket) >= min_bucket_size and bucket[-1] < size:
      buckets.append(bucket[-1])
      bucket = []
    else:
      bucket.append(size)
  if bucket:
    buckets.append(bucket[-1])
  return buckets
