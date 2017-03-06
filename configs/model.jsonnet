{
  architecture: {
      share_word_embeddings: false,
      # 0: trigger_chans
      # 1: trigger_funcs
      # 2: action_chans
      # 3: action_funcs
      label_groupings: [[0], [1], [2], [3]],
      label_types: [0, 1, 2, 3],
      #label_groupings: [[0]],
      #label_types: [0],
      # Maximum word ID allowed.
      # All IDs bigger than this will be replaced with this ID.
      # Set to -1 to disable.
      max_word_id: 4000,
  },
  label_types: [0],

  # Learning parameters
  optim: {
    batch_size: 32,
    lr_scheduler: {
      name: "exponential_decay",
      learning_rate: 1e-2,
      decay_rate: 0.9,
      decay_steps: 1000,
      staircase: true,
    },
    optimizer: {
      # GradientDescentOptimizer, AdamOptimizer, ...
      #name: "GradientDescentOptimizer",
      name: "AdamOptimizer",
    },
    init: {
      name: "random_uniform_initializer",
      scale: 0.1,
    },
    max_grad_norm: 5,
  },

  # Model settings
  model: {
    name: "rnn",
    use_embedding: true,
    bidirectional: true,
    embedding_size: 50, 
    # types: BasicRNNCell, GRUCell, BasicLSTMCell, LSTMCell
    cell_type: "BasicLSTMCell",
    num_layers: 1,
    num_units: 50,
    keep_prob: 0.5,
    memory_size: 25,
    decoder: "LA"
  },

  # Evaluation settings
  eval: {
    # seconds, epochs, steps
    unit: "epochs",
    freq: 0.5,
    max_unsuccessful_trials: 5,
    relevant_labels: [4, 5]
  }

}