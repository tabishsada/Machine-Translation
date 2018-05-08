class Hyperparams:
    # data
    source_train = 'wmt14/train.de'
    target_train = 'wmt14/train.en'
    source_test = 'wmt14/test.de'
    target_test = 'wmt14/test.en'

    # training
    batch_size = 32
    lr = 0.0001 # learning rate.
    logdir = 'logdir'

    maxlen = 10
    min_cnt = 20 # words whose occurred less than min_cnt are encoded as <UNK>.
    hidden_units = 512 # alias = C
    num_blocks = 6 # number of encoder/decoder blocks
    num_epochs = 20
    num_heads = 8
    dropout_rate = 0.1
    sinusoid = False 
