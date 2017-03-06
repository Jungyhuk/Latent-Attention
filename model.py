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


label_names = ['trigger_chans', 'trigger_funcs', 'action_chans',
               'action_funcs']


class keydefaultdict(collections.defaultdict):
  def __missing__(self, key):
    if self.default_factory is None:
      raise KeyError(key)
    else:
      ret = self[key] = self.default_factory(key)
      return ret


def without_name(d):
  d = d.copy()
  del d['name']
  return d


def variable_summaries(var):
  name = var.name
  with tf.name_scope("summaries"):
    mean = tf.reduce_mean(var)
    tf.scalar_summary('vars-mean/' + name, mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
    tf.scalar_summary('vars-sttdev/' + name, stddev)
    tf.scalar_summary('vars-max/' + name, tf.reduce_max(var))
    tf.scalar_summary('vars-min/' + name, tf.reduce_min(var))
    tf.histogram_summary(name, var)


def make_train_op(global_step, loss, optim_config):
  lr = getattr(tf.train, optim_config['lr_scheduler']['name'])(
      global_step=global_step,
      **without_name(optim_config['lr_scheduler']))
  tf.scalar_summary('lr', lr)
  tvars = tf.trainable_variables()
  #  for var in tvars:
  #    variable_summaries(var)
  grads = tf.gradients(loss, tvars)
  if optim_config['max_grad_norm'] > 0:
    grads, grads_norm = tf.clip_by_global_norm(grads,
                                               optim_config['max_grad_norm'])
  else:
    grads_norm = tf.global_norm(grads)
  tf.scalar_summary('grads/norm', grads_norm)
  optimizer = getattr(tf.train, optim_config['optimizer']['name'])(
      learning_rate=lr,
      **without_name(optim_config['optimizer']))
  return optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)

class IFTTTModel(object):
  def __init__(self, optim_config, model_config, arch_config, num_labels,
               label_types, vocab_size, memory_size, is_training):
    batch_size = optim_config['batch_size']
    self.ids = tf.placeholder(tf.int32, name='input',
                              shape=[batch_size, memory_size])
    self.ids_lengths = tf.placeholder(
        tf.int32, name='input_lengths', shape=[batch_size])
    self.labels = tf.placeholder(tf.int32,
                                 name='labels',
                                 shape=[batch_size, len(num_labels)])
    self.expanded_labels = tf.placeholder(
        tf.int32,
        name='expanded_labels',
        # batch size x seq length x label types
        shape=[optim_config['batch_size'], memory_size, len(num_labels)])

    with tf.name_scope("input_length_summaries"):
        max_length = tf.reduce_max(self.ids_lengths)
        tf.scalar_summary("input_length/max", max_length)
        tf.scalar_summary("input_length/min", tf.reduce_min(self.ids_lengths))
        mean_length = tf.reduce_mean(tf.cast(self.ids_lengths, tf.float32))
        tf.scalar_summary("input_length/mean", mean_length)
        tf.scalar_summary("input_length/fill_ratio",
                          mean_length / tf.cast(max_length, tf.float32))

    ids = tf.minimum(vocab_size - 1, self.ids, name='make_unk')
    if arch_config:
      label_groupings = arch_config['label_groupings']

      if arch_config['share_word_embeddings']:
        embedding = tf.get_variable(
            'embedding_for_' + '_'.join(str(i) for i in label_types),
            [vocab_size, model_config['embedding_size']])
        rnn_inputs = [tf.nn.embedding_lookup(embedding, ids)
                      ] * len(label_groupings)
      else:
        rnn_inputs = []
        for grouping in label_groupings:  #enumerate(label_groupings):
          embedding = tf.get_variable(
              'embedding_for_' + '_'.join(str(i) for i in grouping),
              [vocab_size, model_config['embedding_size']])
          rnn_inputs.append(tf.nn.embedding_lookup(embedding, ids))

      outputs = []
      for grouping, rnn_input in zip(label_groupings, rnn_inputs):
        with tf.variable_scope('labels_' + '_'.join(str(i) for i in grouping)):
          if model_config['name'] == 'rnn':
            outputs.append(tf_utils.models.rnn(
              rnn_input,
              self.ids_lengths,
              getattr(tf.nn.rnn_cell, model_config['cell_type']),
              int(model_config['num_layers']),
              int(model_config['num_units']),
              model_config['keep_prob'],
              is_training,
              bidirectional=model_config['bidirectional'],
              debug=False))
          elif model_config['name'] == 'Dict':
            outputs.append((rnn_input,tf.reduce_sum(rnn_input,1)))

    else:
      label_groupings = [label_types]
      embedding = tf.get_variable('embedding',
                                  [vocab_size, model_config['embedding_size']])
      rnn_input = tf.nn.embedding_lookup(embedding, ids)
      if model_config['name'] == 'rnn':
        outputs = [tf_utils.models.rnn(
          rnn_input,
          self.ids_lengths,
          getattr(tf.nn.rnn_cell, model_config['cell_type']),
          int(model_config['num_layers']),
          int(model_config['num_units']),
          model_config['keep_prob'],
          is_training,
          bidirectional=model_config['bidirectional'],
          debug=False)]
      else:
        outputs = [(rnn_input,tf.reduce_sum(rnn_input,1))]

    losses = []
    all_preds = []
    self.all_probs = []
    decoder_type = model_config.get('decoder', 'standard')


    for i, num_classes in enumerate(num_labels):
        with tf.variable_scope('label_{}'.format(label_types[i])):
            for (states, output), group in zip(outputs, label_groupings):
                if label_types[i] in group:
                    break
            else:
                raise ValueError('label_types[i] {} not found in {}'.format(
                    label_types[i], label_groupings))

            if decoder_type in ['LA', 'attention']:
                if model_config['name'].find('rnn') != -1:
                    vec_size = model_config['embedding_size'] * 2
                else:
                    vec_size = model_config['embedding_size']

                PREP = tf.get_variable("PREP", [1, vec_size])
                softmax_w = tf.get_variable(
                    "softmax_w", [vec_size, num_classes],
                    initializer = tf.contrib.layers.xavier_initializer())
                TA = tf.get_variable("TA", [memory_size, vec_size])
                m = states + TA

                if decoder_type == 'LA':
                    B = tf.get_variable("B", [vec_size, memory_size])
                    m_t = tf.reshape(m,
                                     [optim_config['batch_size'] * memory_size,
                                      vec_size])
                    d_t = tf.matmul(m_t, B)
                    d_softmax = tf.nn.softmax(d_t)
                    d = tf.reshape(d_softmax, [optim_config['batch_size'],
                                               memory_size,
                                               memory_size])
                    dotted_prep = tf.reduce_sum(states * PREP, 2)
                else: # 'attention'
                    dotted_prep = tf.reduce_sum(m * PREP, 1)

                probs_prep = tf.nn.softmax(dotted_prep)
                preps = []
                preps.append(probs_prep)
                for _ in xrange(1):
                    probs_prep = preps[-1]
                    if decoder_type == 'LA':
                        probs_prep_temp = tf.expand_dims(probs_prep, -1)
                        probs_temp = tf.batch_matmul(d, probs_prep_temp)
                        probs = tf.squeeze(probs_temp)
                        output_probs = tf.nn.l2_normalize(probs, 1)
                    else: # 'attention'
                        probs_prep_temp = tf.transpose(
                            tf.expand_dims(probs_prep, -1), [0, 2, 1])
                        dotted = tf.reduce_sum(m * probs_prep_temp, 2)
                        output_probs = tf.nn.softmax(dotted)
                    preps.append(output_probs)

                probs_temp = tf.expand_dims(preps[-1], 1)
                c_temp = tf.transpose(m, [0, 2, 1])
                o_k = tf.reduce_sum(c_temp * probs_temp, 2)
                logits = tf.matmul(o_k, softmax_w)
            else:
                w = tf.get_variable('softmax_w', [output.get_shape()[1],
                                              num_classes])
                b = tf.get_variable('softmax_b', [num_classes])
                logits = tf.nn.xw_plus_b(output, w, b)

            preds = tf.cast(tf.argmax(logits, 1), tf.int32)
            all_preds.append(preds)
            probs = tf.nn.softmax(logits)
            self.all_probs.append(probs)
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits, self.labels[:, i]))
            losses.append(loss)
            tf.scalar_summary(
                'train_acc/{}'.format(label_names[label_types[i]]),
                tf.reduce_mean(tf.cast(
                    tf.equal(preds, self.labels[:, i]), tf.float32)))

    self.preds = tf.transpose(tf.pack(all_preds))
    self.loss = tf.add_n(losses)

    tf.scalar_summary('loss', self.loss)

    self.global_step = tf.Variable(0, trainable=False, name='global_step')
    if is_training:
      self.train_op = make_train_op(self.global_step, self.loss, optim_config)

    self.summaries = tf.merge_all_summaries()
