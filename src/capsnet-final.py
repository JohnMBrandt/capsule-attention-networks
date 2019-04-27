
# coding: utf-8

# In[1]:


import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# In[2]:


import tensorflow as tf
import pickle
import numpy as np
import os

def load_obj(name, direct):
    'Helper function using pickle to save and load objects'
    with open("data/" + direct + "/" + name, "rb") as f:
        d =  pickle.load(f, encoding='latin1')
        return d
    
def chunks(l, n):
    chunked = np.empty([3, 3, 6])
    for i in range(0, 54, n):
        reshaped = np.reshape(l[i:i + n], (3, 3))
        chunked[0:3, 0:3, i//n] = reshaped
    return chunked

def data_generator(files, direct):
    while True:
        i = 0
        for file in files:
            i += 1
            f = load_obj(file, direct)
            x_begin = f[0][:, :, 1:]
            x_end = np.empty([500, 26, 3, 3, 6])
            for i in range(len(x_begin)):               # for i in 0:500
                for j in range(len(x_begin[i])):        # for j in 0:26
                    chk = chunks(x_begin[i, j, :], 9)
                    x_end[i, j, 0:3, 0:3, 0:6] = chk
            yield x_end, f[1], f[2]


# In[3]:


from sklearn.utils.class_weight import compute_class_weight
weights_all = []
for i in range(0, 50):  
    x, y, seq_lengths = (next(data_generator(os.listdir("data/train/"), "train")))
    tst = np.argmax(np.reshape(y, (500*26, 23)), 1)
    lengths_transposed = np.expand_dims(seq_lengths, 1)
    rng = np.arange(0, 26, 1)
    range_row = np.expand_dims(rng, 0)
    masked = np.less(range_row, lengths_transposed)
    masked = np.reshape(masked, (500*26))
    masked = masked.astype(int)
    weights_all.append(tst)
weights_all = [item for sublist in weights_all for item in sublist]
weights_crop = [x for x in weights_all if x > 3]
weight = compute_class_weight(classes = np.unique(weights_all), y = weights_all, class_weight ='balanced')
weight_crop = compute_class_weight(classes = np.unique(weights_crop), y = weights_crop, class_weight = 'balanced')
print(weight)
print(weight_crop)


# In[4]:


import random

tf.reset_default_graph()

batch_size = 100
lstm_size = 220
n_steps = 26
num_input = 54
learning_rate = 0.0025
dim_capsule = 12
n_classes = 23

args_dict = {'Batch size' : batch_size, 
            'LSTM size' : lstm_size,
            'Dim capsule' : dim_capsule,
            'Learning rate' : learning_rate}

# [batch, in_depth, in_height, in_width, in_channels].
X = tf.placeholder(tf.float32, shape=(batch_size, n_steps, 3, 3, 6))
y = tf.placeholder(tf.float32, shape=(batch_size, n_steps, n_classes))
seq_lengths = tf.placeholder(tf.int32, [batch_size], name="seq_lengths")

def create_weights(shape, initializer = 'xavier'):
    initializer = tf.contrib.layers.xavier_initializer()
    if initializer != 'xavier':
        initializer = initializers.get('glorot_uniform')
    return(tf.Variable(initializer(shape)))
 
def create_biases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))

def create_convolutional_layer(input):  
 
    print("Conv input shape: {}".format(input.shape))
    layers = []
    layer0 = tf.layers.conv2d(inputs=input[:, 0, :, :, :],
                         filters=64,
                         kernel_size = 1,
                         strides=[1, 1],
                         padding='valid',
                              activation = tf.nn.relu,
                              name = 'convolution1')
    layer0 = tf.reshape(layer0, (-1, 3, 3, 64))
    layers.append(layer0)
    for i in range(1, 26):
    ## Creating the convolutional layer
        layer = tf.layers.conv2d(inputs=input[:, i, :, :, :],
                         filters=64,
                         kernel_size = 1,
                         strides=[1, 1],
                         padding='valid',
                         activation = tf.nn.relu,
                                 reuse = True,
                         name = 'convolution1',
                         )

        layer = tf.reshape(layer, (-1, 3, 3, 64))
        layers.append(layer)
    
    output = tf.stack(layers, axis = 1)
    print("Conv output shape: {}".format(output.shape))
    return output

def create_lstm(x, seq_length):
    print("LSTM input shape: {}".format(x.shape))
    lstm_cell_fw = tf.nn.rnn_cell.BasicLSTMCell(lstm_size, state_is_tuple = True)
    lstm_cell_bw = tf.nn.rnn_cell.BasicLSTMCell(lstm_size, state_is_tuple = True)
    states, final_state = tf.nn.bidirectional_dynamic_rnn(
                                        cell_fw = lstm_cell_fw, 
                                        cell_bw = lstm_cell_bw,
                                        inputs = x, 
                                        dtype = tf.float32,
                                        time_major = False,
                                        sequence_length=seq_length)
    print("LSTM out shape: {}".format(states[0].shape))
    output = tf.concat([states[0], states[1]], axis = 2)
    print("LSTM final shape: {}".format(output.shape))
    return output

def create_deep_lstm(x, seq_length):
    print("Deep LSTM input: {}".format(x.shape))
    with tf.name_scope('RNN'):
        #cell = tf.nn.rnn_cell.LSTMCell(n_rnn_cells)
        #cell = tf.contrib.rnn.DropoutWrapper(cell=cell, output_keep_prob= 1)
        #cell = tf.nn.rnn_cell.MultiRNNCell([cell] * 3)
        def lstm_cell():
            return tf.contrib.rnn.BasicLSTMCell(lstm_size)
        
        cells_fw = [lstm_cell() for _ in range(4)]
        print(cells_fw)
        cells_bw = [lstm_cell() for _ in range(4)]
        #state_fw = cells_fw[0].zero_state(batch_size, tf.float32)
        #state_bw = cells_bw[0].zero_state(batch_size, tf.float32)
        
        outputs, last_states, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
            cells_fw = cells_fw,
            cells_bw = cells_bw,
            inputs = x,
            #initial_states_fw = state_fw,
            #initial_states_bw = state_bw,
            dtype=tf.float32,
            sequence_length=seq_length,
        )
        #stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(2)])
        #state = stacked_lstm.zero_state(batch_size, tf.float32)
        #outputs, last_states = tf.nn.dynamic_rnn(stacked_lstm, x,
        #                                         initial_state=state, sequence_length=seq_length,
        #                                                 time_major=False)
        #output = tf.concat([outputs[0], outputs[1]], axis = 2)
        print("Deep LSTM out: {}".format(outputs.shape))
        print("Deep LSTM concat: {}".format(outputs.shape))

        return outputs

def distributed_dense(x, units, activation, name):
    print("Dense in: {}".format(x.shape))
    dense_layers = []
    for i in range(0, n_steps):
        data = tf.squeeze(x[:, i, :])
        if i == 0:
            dense_l = tf.layers.dense(data,
                                    units,
                                    activation,
                                   name = name,
                                    reuse = False)
        else: 
            dense_l = tf.layers.dense(data,
                                    units,
                                    activation,
                                   name = name,
                                    reuse = True)
        dense_layers.append(dense_l)
    dense_layers = tf.stack(dense_layers, axis = 1)
    print("Dense out: {}".format(dense_layers.shape))
    return dense_layers


# In[5]:


import keras.backend as K
from keras import initializers, layers

def squash(vectors, axis=-1):
    """
    The non-linear activation used in Capsule. It drives the length of a large vector to near 1 and small vector to 0
    :param vectors: some vectors to be squashed, N-dim tensor
    :param axis: the axis to squash
    :return: a Tensor with same shape as input vectors
    """
    s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / K.sqrt(s_squared_norm + K.epsilon())
    return scale * vectors

class CapsuleLayer(layers.Layer):
    """
    The capsule layer. It is similar to Dense layer. Dense layer has `in_num` inputs, each is a scalar, the output of the 
    neuron from the former layer, and it has `out_num` output neurons. CapsuleLayer just expand the output of the neuron
    from scalar to vector. So its input shape = [None, input_num_capsule, input_dim_capsule] and output shape = \
    [None, num_capsule, dim_capsule]. For Dense Layer, input_dim_capsule = dim_capsule = 1.
    
    :param num_capsule: number of capsules in this layer
    :param dim_capsule: dimension of the output vectors of the capsules in this layer
    :param routings: number of iterations for the routing algorithm
    """
    def __init__(self, num_capsule, dim_capsule, routings=3,
                 kernel_initializer='glorot_uniform',
                 **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsule = 23
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_initializer = initializers.get(kernel_initializer)

    def build(self, input_shape):
        assert len(input_shape) >= 3, "The input Tensor should have shape=[None, input_num_capsule, input_dim_capsule]"
        self.input_num_capsule = input_shape[1]
        self.input_dim_capsule = input_shape[2]

        # Transform matrix
        self.W = self.add_weight(shape=[self.num_capsule, self.input_num_capsule,
                                        self.dim_capsule, self.input_dim_capsule],
                                 initializer=self.kernel_initializer,
                                 name='W')

        self.built = True

    def call(self, inputs, training=None):
        # inputs.shape=[None, input_num_capsule, input_dim_capsule]
        # inputs_expand.shape=[None, 1, input_num_capsule, input_dim_capsule]
        inputs_expand = K.expand_dims(inputs, 1)

        # Replicate num_capsule dimension to prepare being multiplied by W
        # inputs_tiled.shape=[None, num_capsule, input_num_capsule, input_dim_capsule]
        inputs_tiled = K.tile(inputs_expand, [1, self.num_capsule, 1, 1])

        # Compute `inputs * W` by scanning inputs_tiled on dimension 0.
        # x.shape=[num_capsule, input_num_capsule, input_dim_capsule]
        # W.shape=[num_capsule, input_num_capsule, dim_capsule, input_dim_capsule]
        # Regard the first two dimensions as `batch` dimension,
        # then matmul: [input_dim_capsule] x [dim_capsule, input_dim_capsule]^T -> [dim_capsule].
        # inputs_hat.shape = [None, num_capsule, input_num_capsule, dim_capsule]
        inputs_hat = K.map_fn(lambda x: K.batch_dot(x, self.W, [2, 3]), elems=inputs_tiled)

        # Begin: Routing algorithm ---------------------------------------------------------------------#
        # The prior for coupling coefficient, initialized as zeros.
        # b.shape = [None, self.num_capsule, self.input_num_capsule].
        b = tf.zeros(shape=[K.shape(inputs_hat)[0], self.num_capsule, self.input_num_capsule])

        assert self.routings > 0, 'The routings should be > 0.'
        for i in range(self.routings):
            # c.shape=[batch_size, num_capsule, input_num_capsule]
            c = tf.nn.softmax(b, dim=1)

            # c.shape =  [batch_size, num_capsule, input_num_capsule]
            # inputs_hat.shape=[None, num_capsule, input_num_capsule, dim_capsule]
            # The first two dimensions as `batch` dimension,
            # then matmal: [input_num_capsule] x [input_num_capsule, dim_capsule] -> [dim_capsule].
            # outputs.shape=[None, num_capsule, dim_capsule]
            outputs = squash(K.batch_dot(c, inputs_hat, [2, 2]))  # [None, 10, 16]

            if i < self.routings - 1:
                # outputs.shape =  [None, num_capsule, dim_capsule]
                # inputs_hat.shape=[None, num_capsule, input_num_capsule, dim_capsule]
                # The first two dimensions as `batch` dimension,
                # then matmal: [dim_capsule] x [input_num_capsule, dim_capsule]^T -> [input_num_capsule].
                # b.shape=[batch_size, num_capsule, input_num_capsule]
                b += K.batch_dot(outputs, inputs_hat, [2, 3])
        # End: Routing algorithm -----------------------------------------------------------------------#

        return outputs

    def compute_output_shape(self, input_shape):
        return tuple([None, self.num_capsule, self.dim_capsule])

    def get_config(self):
        config = {
            'num_capsule': self.num_capsule,
            'dim_capsule': self.dim_capsule,
            'routings': self.routings
        }
        base_config = super(CapsuleLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def PrimaryCap(inputs, dim_capsule, n_channels, kernel_size, strides, padding, i):
    """
    Apply Conv2D `n_channels` times and concatenate all capsules
    :param inputs: 4D tensor, shape=[None, width, height, channels]
    :param dim_capsule: the dim of the output vector of capsule
    :param n_channels: the number of types of capsules
    :return: output tensor, shape=[None, num_capsule, dim_capsule]
    """
    if i == 0:
        output = tf.layers.conv2d(inputs, filters=dim_capsule*n_channels, kernel_size=kernel_size, strides=strides, padding=padding,
                           name='primarycap_conv2d', reuse = None)
    else: 
        output = tf.layers.conv2d(inputs, filters=dim_capsule*n_channels, kernel_size=kernel_size, strides=strides, padding=padding,
                           name='primarycap_conv2d', reuse = True)
    
    outputs = layers.Reshape(target_shape=[-1, dim_capsule], name='primarycap_reshape')(output)
    return layers.Lambda(squash, name='primarycap_squash')(outputs)

def multi_capsule(x):
    n_channels = 64
    caps = CapsuleLayer(num_capsule = n_classes, dim_capsule=dim_capsule, routings=4, name='digitcaps')
    #primary_layers = []
    caps_layers = []
    for i in range(0, n_steps):
        xi = tf.squeeze(x[:, i, :, :, :])
        primarycaps = PrimaryCap(xi, dim_capsule=dim_capsule, n_channels=n_channels, kernel_size=3, strides=1, padding='valid', i = i)
        if i == 0:
            print("Primary cap squeeze shape: {}".format(primarycaps.shape))
        caps_layer = caps(primarycaps) 
        caps_layers.append(caps_layer)
    output = tf.stack(caps_layers, axis = 1)
    #output = tf.reshape(output, (batch_size, 23*8))
    print("Capsule out: {}".format(output.shape))
    return output


def create_multi_attention(inputs, attention_size, time_major=False):
    hidden_size = inputs.shape[2].value
    print("Attention In: {}".format(inputs.shape))
   
    w_omegas, b_omegas, u_omegas = [], [], []
    for i in range(0, 26):
        w_omegas.append(create_weights([hidden_size, attention_size]))
        b_omegas.append(tf.Variable(tf.constant(0.05, shape = [attention_size])))
        u_omegas.append(create_weights([attention_size]))
        
    # Trainable parameters
    layers_all = []
    for i in range(0, 26):  
        v = tf.tanh(tf.tensordot(inputs, w_omegas[i], axes=1) + b_omegas[i])       
        # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
        vu = tf.tensordot(v, u_omegas[i], axes=1, name='vu')  # (B,T) shape\
        alphas = tf.nn.softmax(vu, name='alphas')         # (B,T) shape

        # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
        output = tf.reshape(tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1), (batch_size, 1, hidden_size))
        
        layers_all.append(output)
    output = tf.concat(layers_all, axis = 1)
    print("Attention Out: {}".format(output.shape))
    return output


# In[6]:


from sklearn import metrics
def crop_net(x, attention = True, skip = True):  
    conv1 = create_convolutional_layer(input = x)
    capsule = multi_capsule(conv1)
    capsule_reshape = tf.reshape(capsule, (-1, n_steps, 23*dim_capsule))
    #lstm_out = create_lstm(capsule_reshape, seq_lengths)
    lstm_out = create_deep_lstm(capsule_reshape, seq_lengths)
    attention = create_multi_attention(lstm_out, 100)
    #attention_concat = tf.concat(attention, capsule_reshape, axis = 3)

    dense1 = distributed_dense(attention, 512, tf.nn.relu, "dense1")
    dense2 = distributed_dense(dense1, 23, None, "dense2")
    dense2 = tf.reshape(dense2, (batch_size*26, 23))
    return dense2

def make_confusion_plot(confusion):
    confusion2 = np.zeros((23, 23))
    for i, row in enumerate(confusion):
        confusion2[i] = row / np.maximum(1, np.sum(row))
    plt.clf()
    sns.heatmap(confusion2.T, cmap = plt.cm.Blues)
    plt.show()
    
    
def one_hot(a, num_classes):
    a = np.asarray(a)
    b = np.zeros((len(a), num_classes))
    b[np.arange(len(a)), a] = 1
    return(b)
    
def calc_crop_acc(pred, true, class_weight):
    index = [index for index, value in enumerate(true) if value > 3]
    true_crop = np.asarray([true[i] - 4 for i in index])
    pred_crop = np.asarray([pred[i] - 4 for i in index])
    #true_onehot = one_hot(true_crop, 19)
    #weighted = np.sum(true_onehot * class_weight, axis = 1)
    
    w_field = np.ones(true_crop.shape[0])
    for idx, i in enumerate(np.bincount(true_crop)):
        w_field[true_crop == idx] *= (i/float(true_crop.shape[0]))
    accuracy_sc = metrics.accuracy_score(true_crop, pred_crop, sample_weight=w_field)
    
    true_ar = np.asarray(true)
    pred_ar = np.asarray(pred)
    w_all = np.ones(true_ar.shape[0])
    for idx, i in enumerate(np.bincount(true_ar)):
        w_all[true_ar == idx] *= (i/float(true_ar.shape[0]))
    acc_sc_all = metrics.accuracy_score(true_ar, pred_ar, sample_weight=w_all)
    
    
    print("Weighted P R F S {}".format(metrics.precision_recall_fscore_support(true_crop, pred_crop, average = 'weighted')))
    #print("Balanced crop accuracy {}".format(metrics.accuracy_score(true_crop, pred_crop, True, weighted)))
    print("Crop accuracy: {}".format(np.mean(np.equal(true_crop, pred_crop))))
    print("Weighted crop accuracy {}".format(accuracy_sc))
    print("weighted all accuracy: {} ".format(acc_sc_all))


# In[7]:


# Construct model
logits = crop_net(X)
print("Logits shape: {}".format(logits.shape))

y_pred = tf.nn.softmax(logits)
y_out = tf.reshape(y, (batch_size*26, 23))

lengths_transposed = tf.expand_dims(seq_lengths, 1)
rng = tf.range(0, 26, 1)
range_row = tf.expand_dims(rng, 0)
mask_boolean = tf.reshape(tf.less(range_row, lengths_transposed), (batch_size*26,))
mask = tf.cast(mask_boolean, tf.float32)

class_weights = tf.constant(weight, dtype = tf.float32) 
print("Class weights: {}".format(weight))
weight_map = tf.multiply(y_out, class_weights)
weight_map = tf.reduce_sum(weight_map, axis=1)

cross_entropy_matrix = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels= y_out)
print("Cross entropy size: {}".format(cross_entropy_matrix.shape))
weighted_losses = cross_entropy_matrix * mask # weight_map
cross_entropy = tf.reduce_sum(weighted_losses) / tf.cast(tf.reduce_sum(seq_lengths), tf.float32)
print("Final loss shape: {}".format(cross_entropy.shape))

# Define loss and optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(weighted_losses)

y_pred = tf.boolean_mask(tf.argmax(tf.nn.softmax(logits), 1), mask_boolean)
y_true = tf.boolean_mask(tf.argmax(y_out, 1), mask_boolean)
accuracy_vector = tf.equal(y_pred, y_true)
accuracy = tf.reduce_mean(tf.cast(accuracy_vector, tf.float32))


# In[ ]:


saver = tf.train.Saver(max_to_keep = 2)
#import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import sklearn
#get_ipython().run_line_magic('matplotlib', 'inline')
    
for key, value in args_dict.items():
    print(key + ": " + str(value))

with tf.Session() as sess:
    #saver.restore(sess, "/tmp/model.ckpt")
    #print("Model restored.")
    tf.global_variables_initializer().run()
    losses, accuracies = [], []
    max_step = 50000
    step = 0
    datagen = data_generator(os.listdir("data/train/"), direct = 'train')
    test_gen = data_generator(os.listdir("data/test/"), direct = 'test')
    eval_gen = data_generator(os.listdir("data/eval/"), direct = 'eval')
    while step < max_step:
        # save_path = saver.save(sess, os.getcwd() + "/model.ckpt")
        X_batch, y_batch, seq_len = next(datagen)
        splits = [x for x in range(0, 481, batch_size)]
        for i in splits:
                step += 1
                minibatch_x = X_batch[i : i + batch_size]
                minibatch_y = y_batch[i : i + batch_size]
                minibatch_len = seq_len[i : i + batch_size]
                feed_dict = {X: minibatch_x, y: minibatch_y, seq_lengths : minibatch_len} 
                _, loss_value, acc = sess.run([train_op, cross_entropy, accuracy], feed_dict)
                accuracies.append(acc)
                losses.append(loss_value)
                if step % 25 == 0:
                    print("Batch {}: Training accuracy: {} Loss: {}".format(step, np.mean(accuracies), np.mean(losses)))
                    losses, accuracies = [], []
        if step % 250 == 0:
            preds, trues, test_acc = [], [], []
            for i in range(0, 3):
                X_batch, y_batch, seq_len = next(test_gen)
                splits = [x for x in range(0, 481, batch_size)]
                for i in splits:
                    minibatch_x = X_batch[i : i + batch_size]
                    minibatch_y = y_batch[i : i + batch_size]
                    minibatch_len = seq_len[i : i + batch_size]
                    feed_dict={X: minibatch_x, y: minibatch_y, seq_lengths : minibatch_len}
                    test_acc.append(sess.run(accuracy, feed_dict = feed_dict))
                    pred, true = sess.run([y_pred, y_true], feed_dict)
                    preds.append(pred)
                    trues.append(true)
            pred = [item for sublist in preds for item in sublist]
            true = [item for sublist in trues for item in sublist]
            print("Test accuracy: {}".format(np.mean(test_acc)))
            calc_crop_acc(pred, true, weight_crop)
            confusion = tf.confusion_matrix(labels=true, predictions=pred, num_classes=23).eval(session=sess)
            #make_confusion_plot(confusion)
        if step % 10000 == 0:
            save_path = saver.save(sess, "models/model.ckpt")
            print("Model saved in path: %s" % save_path)
            preds, trues, test_acc = [], [], []
            for i in range(0, 115):
                X_batch, y_batch, seq_len = next(eval_gen)
                splits = [x for x in range(0, 481, batch_size)]
                for i in splits:
                    minibatch_x = X_batch[i : i + batch_size]
                    minibatch_y = y_batch[i : i + batch_size]
                    minibatch_len = seq_len[i : i + batch_size]
                    feed_dict={X: minibatch_x, y: minibatch_y, seq_lengths : minibatch_len}
                    test_acc.append(sess.run(accuracy, feed_dict = feed_dict))
                    pred, true = sess.run([y_pred, y_true], feed_dict)
                    preds.append(pred)
                    trues.append(true)
            pred = [item for sublist in preds for item in sublist]
            true = [item for sublist in trues for item in sublist]
            print("Evaluation accuracy: {}".format(np.mean(test_acc)))
            calc_crop_acc(pred, true, weight_crop)
            confusion = tf.confusion_matrix(labels=true, predictions=pred, num_classes=23).eval(session=sess)
            #make_confusion_plot(confusion)


# In[8]:


with tf.Session() as sess:
    tf.global_variables_initializer().run()
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        print(variable)
        print(shape)
        print(len(shape))
        variable_parameters = 1
        for dim in shape:
            print(dim)
            variable_parameters *= dim.value
        print(variable_parameters)
        total_parameters += variable_parameters
    print(total_parameters)

