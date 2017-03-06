import numpy as np
import tensorflow as tf
import utils

def rnn(inputs,
            input_lengths,
            cell_type,
            num_layers,
            num_units,
            keep_prob,
            is_training,
            bidirectional=False,
            debug=False,
            regular_output=False):
  # inputs: batch x time x depth

  assert num_layers >= 1

  need_tuple_state = cell_type in (tf.nn.rnn_cell.BasicLSTMCell,
                                   tf.nn.rnn_cell.LSTMCell)

  if need_tuple_state:
    cell = cell_type(num_units, state_is_tuple=True)
  else:
    cell = cell_type(num_units)

  if is_training and keep_prob < 1:
    cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)
  if bidirectional:
    input_lengths_64 = tf.cast(input_lengths, tf.int64)
    prev_layer_fwd = inputs
    prev_layer_rev = tf.reverse_sequence(inputs, input_lengths_64, 1)
    for i in xrange(num_layers):
      with tf.variable_scope("Layer%d" % i):
        with tf.variable_scope("Fwd"):
          outputs_fwd, final_state_fwd = tf.nn.dynamic_rnn(cell,
                                                           prev_layer_fwd,
                                                           input_lengths,
                                                           dtype=tf.float32)
        with tf.variable_scope("Rev"):
          outputs_rev, final_state_rev = tf.nn.dynamic_rnn(cell,
                                                           prev_layer_rev,
                                                           input_lengths,
                                                           dtype=tf.float32)

        outputs_rev = tf.reverse_sequence(outputs_rev, input_lengths_64, 1)
        prev_layer_fwd = tf.concat(2, [outputs_fwd, outputs_rev])
        prev_layer_rev = tf.reverse_sequence(prev_layer_fwd, input_lengths_64,
                                             1)
    if regular_output:
      return prev_layer_fwd, final_state_fwd + final_state_rev

    if need_tuple_state:
      final_state_fwd = final_state_fwd[1]
      final_state_fwd.set_shape([inputs.get_shape()[0], cell.state_size[1]])
      final_state_rev = final_state_rev[1]
      final_state_rev.set_shape([inputs.get_shape()[0], cell.state_size[1]])
    else:
      final_state_fwd.set_shape([inputs.get_shape()[0], cell.state_size])
      final_state_rev.set_shape([inputs.get_shape()[0], cell.state_size])

    final_output = tf.concat(1, [final_state_fwd, final_state_rev])
    return prev_layer_fwd, final_output

  # Not bidirectional
  for i in xrange(num_layers):
    prev_layer = inputs
    with tf.variable_scope("Layer%d" % i):
      outputs, final_state = tf.nn.dynamic_rnn(
          cell, prev_layer, input_lengths,
          dtype=tf.float32)
      prev_layer = outputs

  #if num_layers > 1:
  #  cell = tf.nn.rnn_cell.MultiRNNCell(
  #      [cell] * (num_layers),
  #      state_is_tuple=need_tuple_state)
  #if debug:
  #  inputs = utils.tf_print_shape(inputs,
  #                                message='{} RNN input shape: '.format(
  #                                    tf.get_default_graph()._name_stack))

  #if need_tuple_state and num_layers > 1:
  #  # Work around bug with MultiRNNCell and tuple states
  #  initial_state = tuple(tuple(tf.zeros(
  #      tf.pack([tf.shape(inputs)[0], s]),
  #      dtype=tf.float32) for s in sizes) for sizes in cell.state_size)
  #else:
  #  initial_state = None

  #outputs, final_state = tf.nn.dynamic_rnn(cell,
  #                                         inputs,
  #                                         input_lengths,
  #                                         initial_state=initial_state,
  #                                         dtype=tf.float32)
  if regular_output:
    return outputs, final_state

  if need_tuple_state:
    final_state[1].set_shape([inputs.get_shape()[0], cell.state_size[1]])
    return final_state[1]
  else:
    final_state.set_shape([inputs.get_shape()[0], cell.state_size])
    return final_state
