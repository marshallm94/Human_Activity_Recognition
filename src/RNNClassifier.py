from datetime import datetime as dt
import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator
from sklearn.externals import joblib

from sklearn.model_selection import train_test_split


class RNNClassifier(BaseEstimator):
    '''
    A (in progress) class for a (fully-connected) Recurrent Neural Network.
    '''
    def __init__(self, restore=False, restoration_dir='tmp',
                 model_name='final_model'):

        self.restoration_dir = restoration_dir
        self.model_name = model_name
        if restore:
            self._load_prediction_attributes()

    def fit(self, X, y, hidden_layer_architecture, n_outputs, n_steps,
            learning_rate, batch_size, n_epochs, verbose=True,
            log_training=True, restore=False):
        
        self.n_steps = n_steps
        self.hidden_layer_architecture = hidden_layer_architecture
        self.n_outputs = n_outputs

        # X should have shape = (n_observations, n_timesteps, n_attributes)
        X = self._reshape_X(X)
        # TODO: check if y already is int - if it is, continue, if it isn't
        # encode y
        y = self._encode_y(y)
        self.n_inputs = X.shape[2]

        # construction phase
        X_, y_, logits = self.build_network()

        # training operations
        with tf.name_scope('training') as scope:
            with tf.name_scope('loss') as scope:
                xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_,
                                                                          logits=logits)
                loss = tf.reduce_mean(xentropy, name='loss')

                optimizer = tf.train.GradientDescentOptimizer(learning_rate)
                training_op = optimizer.minimize(loss)

            with tf.name_scope('eval'):
                correct = tf.nn.in_top_k(predictions=logits,
                                         targets=y_,
                                         k=1)
                accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

        if log_training:
            loss_summary = tf.summary.scalar('CrossEntropy', loss)
            file_writer = tf.summary.FileWriter(self.log_directory(),
                                                tf.get_default_graph())

        # execution phase
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        n_batches = int(np.ceil(X.shape[0] / batch_size))

        with tf.Session() as sess:
            init.run()
            for epoch in range(n_epochs):
                for batch_index in range(n_batches):
                    X_batch, y_batch = self.fetch_batch(epoch,
                                                        batch_index,
                                                        batch_size,
                                                        X,
                                                        y)

                    feed_dict = {X_: X_batch, y_: y_batch}

                    if log_training:
                        if batch_index % 10 == 0:
                            loss_str = loss_summary.eval(feed_dict=feed_dict)
                            # serves as an "index" into the training loops
                            step = epoch * n_batches + batch_index
                            file_writer.add_summary(loss_str, step)

                    sess.run(training_op, feed_dict=feed_dict)

                if epoch % 10 == 0:
                    if verbose:
                        loss_train = loss.eval(feed_dict=feed_dict)
                        acc_train = accuracy.eval(feed_dict=feed_dict)
                        print(f"Epoch {epoch} | Loss = {loss_train} | Accuracy = {acc_train}")


            filepath = f"./{self.restoration_dir}/{self.model_name}.ckpt"
            save_path = saver.save(sess, filepath)
            file_writer.close()
            self._save_prediction_attributes()

            if verbose:
                print("\nTraining Complete.\n")
                print(f"Network architecture saved to {filepath + '.meta'}")
                print(f"Variable names saved to {filepath + '.index'}")
                print(f"Varible values (i.e. weights/biases) saved to {filepath + '.data-00000-of-00001'}")
                print(f"Attributes needed for prediction saved to {self.restoration_dir}")

    def _reshape_X(self, X):

        if len(X.shape) == 2:
            return X.reshape(X.shape[0], self.n_steps, X.shape[1])
        elif len(X.shape) == 3:
            return X

    def _encode_y(self, y):

        y_classes = np.unique(y)
        encodings = np.arange(len(y_classes))

        self.y_encoding = {y_classes[i]: i for i in encodings}
        for class_, encoding in self.y_encoding.items():
            y[y==class_] = np.int64(encoding)

        return y.reshape(len(y))

    def _save_prediction_attributes(self):

        joblib.dump(self.y_encoding, f"{self.restoration_dir}/y_encoding.obj")
        joblib.dump(self.n_steps, f"{self.restoration_dir}/n_steps.obj")

    def _load_prediction_attributes(self):

        self.y_encoding = joblib.load(f"{self.restoration_dir}/y_encoding.obj")
        self.n_steps = joblib.load(f"{self.restoration_dir}/n_steps.obj")

    def build_network(self):

        with tf.name_scope('network') as scope:

            # define input layer
            with tf.name_scope('input_layer') as scope:
                X_ = tf.placeholder(tf.float32,
                                    shape=(None, self.n_steps, self.n_inputs),
                                    name='X')

                y_ = tf.placeholder(tf.int64,
                                    shape=(None),
                                    name='y')

            # create recurrent hidden layers
            layers = []
            for layer_id, n_neurons in enumerate(self.hidden_layer_architecture):

                hidden_layer = tf.contrib.rnn.BasicRNNCell(n_neurons,
                                                           activation=tf.nn.tanh,
                                                           name=f'hidden_layer_{layer_id}')
                layers.append(hidden_layer)

            recurrent_hidden_layers = tf.contrib.rnn.MultiRNNCell(layers)
            output, state = tf.nn.dynamic_rnn(recurrent_hidden_layers,
                                                X_, dtype=tf.float32)

            # defining output layer that uses the last output of the last 
            # recurrent layer in the network as its input (output[:,-1,:]).
            # This will return probabilites (need to use argmax to get the actual
            # prediction - the one with the highest probability
            logits = self.neuron_layer(output[:, -1, :],
                                       self.n_outputs,
                                       name='outputs')

        return X_, y_, logits

    def predict(self, X, decode=True):
        
        X = self._reshape_X(X)

        with tf.Session() as sess:
            model = f"{self.restoration_dir}/{self.model_name}.ckpt"

            # loads the network structure to the default graph
            uploader = tf.train.import_meta_graph(f"{model + '.meta'}")

            # loads the variable names and values into the namespace (see 
            # self.fit() method for explanation)
            uploader.restore(sess,
                             tf.train.latest_checkpoint(f"{self.restoration_dir}/"))

            graph = tf.get_default_graph()

            # input tensor for network
            X_ = graph.get_tensor_by_name("network/input_layer/X:0")
            # output operation of network
            logits = graph.get_tensor_by_name("network/outputs/add:0")

            # probabilites of each class
            y_proba = logits.eval(feed_dict={X_: X})
            # choosing class with highest probability
            y_hat = np.argmax(y_proba, axis=1)

        if not decode:

            return y_hat

        elif decode:

            decoded_y_hat = y_hat.astype(str)
            for k, v in self.y_encoding.items():
                decoded_y_hat[decoded_y_hat==str(v)] = k

            return decoded_y_hat

    # manual version of tf.layers.dense()
    def neuron_layer(self, X, n_neurons, name, activation=None):

        # defining scope for TensorBoard
        with tf.name_scope(name):
            n_inputs = int(X.get_shape()[1])
            stddev = 2 / np.sqrt(n_inputs + n_neurons)
            # creates a n_inputs x n_neurons matrix with random values from
            # a truncated normal distribution ("truncated" meaning values that
            # are more than two standard deviations from the mean are dropped
            # and re-picked. Mean default=0.
            init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)

            w = tf.Variable(init,
                            name='weights')
            b = tf.Variable(tf.zeros([n_neurons]),
                            name='bias')
            Z = tf.matmul(X, w) + b
            
            if activation is not None:
                return activation(Z)
            elif activation is None:
                return Z

    def fetch_batch(self, epoch, batch_index, batch_size, X, y,
                    sequential=False):

        if sequential:
            indices = range(batch_index*batch_size,
                            min(batch_index*batch_size+batch_size,len(y)))
        elif not sequential:
            # Note to self: using the not sequential (aka random) sampling for
            # batches does NOT ensure that all observations in the training
            # set will be used. Further, note that np.random.randint() doesn't
            # ensure that all the numbers returned are unique. In other words,
            # regardless of the seed, there is a chance that indices will contain
            # the same observation(s) more than once.
            np.random.seed(epoch * batch_index + batch_size)

            indices = np.random.randint(X.shape[0], size=batch_size)

        X_batch = X[indices]
        y_batch = y[indices]

        return X_batch, y_batch

    def log_directory(self):

        now = dt.utcnow().strftime("%Y-%m-%d_%H:%M:%S")
        root_logdir = 'training_logs'
        logdir = f"{root_logdir}/run-{now}/"

        return logdir


if __name__ == "__main__":

    # import mnist data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)

    x_train, x_validate, y_train, y_validate = train_test_split(x_train, y_train)

    # network architecture
    hidden_layer_array = [1000, 750, 500]
    n_outputs = len(np.unique(y_train))
    n_steps = 28

    rnn = RNNClassifier()

    # training parameters
    learning_rate = 0.01
    n_epochs = 5
    batch_size = 50

    rnn.fit(X=x_train, 
            y=y_train,
            hidden_layer_architecture=hidden_layer_array,
            n_outputs=n_outputs,
            n_steps=n_steps,
            learning_rate=learning_rate,
            batch_size=batch_size,
            n_epochs=n_epochs)

    rnn = RNNClassifier(restore=True)

    y_hat_validate = rnn.predict(x_validate).astype(int)
    validation_acc = np.mean(y_hat_validate == y_validate)

    print(f"\nValidation Accuracy = {validation_acc}\n")

    y_hat = rnn.predict(x_test).astype(int)
    testing_acc = np.mean(y_hat == y_test)

    print(f"\nTesting Accuracy = {testing_acc}\n")
