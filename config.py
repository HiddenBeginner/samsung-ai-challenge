class TrainConfig:
    dir_data = './outputs/rgcn'   # The directory where preprocessed data were stored
    seed = 1                      # For reproducing

    # Model architecture
    num_node_features = 13        # The number of unique atoms
    node_embedding_dim = 256      # The dimension of encoded atom
    hidden_channels = 8 * [256]   # The number of channels of each RGCN layer
    hidden_dims = [1024, 512]     # The number of nodes of each MLP layer in readout phase
    dropout = 0.3                 # The rate of dropout

    # Training
    lr = 0.0001                   # Learning rate
    n_epochs = 300                # The number of epochs
    batch_size = 128              # Batch size
