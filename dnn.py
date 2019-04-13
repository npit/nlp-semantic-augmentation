from os.path import join, exists, basename
from sklearn.exceptions import UndefinedMetricWarning
import warnings
from os import makedirs
import numpy as np
from classifier import Classifier
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from keras.layers import LSTM as keras_lstm
from keras import callbacks
from utils import info, debug, error, write_pickled, one_hot
# from keras import backend
# import tensorflow as tf

warnings.simplefilter(action='ignore', category=UndefinedMetricWarning)


class DNN(Classifier):
    sequence_length = None

    do_multilabel = False
    train_embeddings_params = []
    do_folds = False
    do_validate_portion = False
    early_stopping = None

    model_paths = []

    def __init__(self):
        Classifier.__init__(self)
        pass

    def __str__(self):
        return "name:{} hidden dim:{} nlayers:{} sequence len:{}".format(
            self.name, self.hidden, self.layers, self.sequence_length)

    # define useful keras callbacks for the training process
    def get_callbacks(self):
        self.callbacks = []
        [makedirs(x, exist_ok=True) for x in [self.results_folder, self.models_folder]]

        # model saving with early stoppingtch_si
        self.model_path = self.get_current_model_path()
        weights_path = self.model_path

        # weights_path = os.path.join(models_folder,"{}_fold_{}_".format(self.name, self.fold_index) + "ep_{epoch:02d}_valloss_{val_loss:.2f}.hdf5")
        self.model_saver = callbacks.ModelCheckpoint(weights_path, monitor='val_loss', verbose=0,
                                                     save_best_only=self.validation_exists,
                                                     save_weights_only=False,
                                                     mode='auto', period=1)
        self.callbacks.append(self.model_saver)
        if self.early_stopping_patience and self.validation_exists:
            self.early_stopping = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=self.early_stopping_patience, verbose=0,
                                                          mode='auto', baseline=None, restore_best_weights=False)
            self.callbacks.append(self.early_stopping)

        # stop on NaN
        self.nan_terminator = callbacks.TerminateOnNaN()
        self.callbacks.append(self.nan_terminator)
        # learning rate modifier at loss function plateaus
        if self.validation_exists:
            self.lr_reducer = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                                          patience=10, verbose=0, mode='auto',
                                                          min_delta=0.0001, cooldown=0, min_lr=0)
            self.callbacks.append(self.lr_reducer)
        # logging
        train_csv_logfile = join(self.results_folder, basename(self.get_current_model_path()) + "train.csv")
        self.csv_logger = callbacks.CSVLogger(train_csv_logfile, separator=',', append=False)
        self.callbacks.append(self.csv_logger)
        return self.callbacks

    # to preliminary work
    def make(self, representation, dataset):

        # initialize rng
        # for 100% determinism, you may need to enforce CPU single-threading
        # tf.set_random_seed(self.seed)
        # session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
        # sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
        # backend.set_session(sess)
        Classifier.make(self, representation, dataset)
        pass

    # print information pertaining to early stopping
    def report_early_stopping(self):
        if self.validation_exists and self.early_stopping is not None:
            info("Stopped on epoch {}/{}".format(self.early_stopping.stopped_epoch + 1, self.epochs))
            write_pickled(self.model_path + ".early_stopping", self.early_stopping.stopped_epoch)

    # train a model on training & validation data portions
    def train_model(self, trainval_idx):
        # get data chunks
        train_data, train_labels, val_datalabels = self.get_trainval_data(trainval_idx)
        # define the model
        model = self.get_model()
        # train the damn thing!
        debug("Feeding the network train shapes: {} {}".format(train_data.shape, train_labels.shape))
        if val_datalabels is not None:
            debug("Using validation shapes: {} {}".format(*[v.shape if v is not None else "none" for v in val_datalabels]))
        model.fit(train_data, train_labels,
                  batch_size=self.batch_size,
                  epochs=self.epochs,
                  validation_data=val_datalabels,
                  verbose=self.verbosity,
                  callbacks=self.get_callbacks())
        self.report_early_stopping()
        return model

    # evaluate a dnn
    def test_model(self, test_data, model):
        return model.predict(test_data, batch_size=self.batch_size, verbose=self.verbosity)

    # add softmax classification layer
    def add_softmax(self, model, is_first=False):
        if is_first:
            model.add(Dense(self.num_labels, input_shape=self.input_shape, name="dense_classifier"))
        else:
            model.add(Dense(self.num_labels, name="dense_classifier"))

        model.add(Activation('softmax', name="softmax"))
        return model

    def get_model_path(self):
        return self.model_saver.filepath


class MLP(DNN):
    name = "mlp"

    def __init__(self, config):
        self.config = config
        self.hidden = self.config.learner.hidden_dim
        self.layers = self.config.learner.num_layers
        DNN.__init__(self)

    def make(self, representation, dataset):
        info("Building dnn: {}".format(self.name))
        DNN.make(self, representation, dataset)
        self.input_shape = (self.input_dim,)

    # build MLP model
    def get_model(self):
        model = None
        model = Sequential()
        for i in range(self.layers):
            if i == 0:
                model.add(Dense(self.hidden, input_shape=self.input_shape))
                model.add(Activation('relu'))
                model.add(Dropout(0.3))
            else:
                model.add(Dense(self.hidden))
                model.add(Activation('relu'))
                model.add(Dropout(0.3))

        model = DNN.add_softmax(self, model)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model


class LSTM(DNN):
    name = "lstm"

    def __init__(self, config):
        self.config = config
        self.hidden = self.config.learner.hidden_dim
        self.layers = self.config.learner.num_layers
        self.sequence_length = self.config.learner.sequence_length
        if self.sequence_length is None:
            error("Undefined learner sequence length, but required for {}.".format(self.name))
        DNN.__init__(self)

    # make network
    def make(self, representation, dataset):
        info("Building dnn: {}".format(self.name))
        DNN.make(self, representation, dataset)
        # make sure embedding aggregation is compatible
        # with the sequence-based lstm model
        self.input_shape = (self.sequence_length, self.input_dim)
        aggregation = self.config.representation.aggregation
        # sanity checks
        # sequence-based aggregation
        if aggregation not in ["pad"]:
            error("Aggregation [{}] incompatible with [{}] model.".format(aggregation, self.name))

        # non constant instance element num
        set_instance_lengths = [set(x) for x in representation.get_elements_per_instance()]
        if any([len(x) != 1 for x in set_instance_lengths]):
            error("[{}] needs a constant number of elements per instance per dataset, but got lengths: {}".format(self.name, set_instance_lengths))
        # non-unity elements per instance
        unit_instance_indexes = [[i for i in range(len(x)) if x[i] <= 1] for x in representation.get_elements_per_instance()]
        if any(unit_instance_indexes):
            error("[{}] not compatible with unit instance indexes: {}.".format(self.name, unit_instance_indexes))
        # sequence length data / label matching
        if self.num_train != self.num_train_labels and (self.num_train != self.sequence_length * self.num_train_labels):
            error("Irreconcilable lengths of training data and labels: {}, {} with learner sequence length of {}.".
                  format(self.num_train, self.num_train_labels, self.sequence_length))
        if self.num_test != self.num_test_labels and (self.num_test != self.sequence_length * self.num_test_labels):
            error("Irreconcilable lengths of test data and labels: {}, {} with learner sequence length of {}.".
                  format(self.num_test, self.num_test_labels, self.sequence_length))

    # fetch sequence lstm fold data
    def get_fold_data(self, data, labels=None, data_idx=None, label_idx=None):
        # handle indexes by parent's function
        x, y = DNN.get_fold_data(self, data, labels, data_idx, label_idx)
        # reshape input data to num_docs x vec_dim x seq_len
        if not self.do_train_embeddings:
            x = np.reshape(x, (-1, self.sequence_length, self.input_dim))
        return x, y

    # preprocess input
    def process_input(self, data):
        return np.reshape(data, (-1, self.sequence_length, self.input_dim))

    # build the lstm model
    def get_model(self):
        model = Sequential()
        for i in range(self.layers):
            if self.layers == 1:
                # one and only layer
                model.add(keras_lstm(self.hidden, input_shape=self.input_shape))
            elif i == 0 and self.layers > 1:
                # first layer, more follow
                model.add(keras_lstm(self.hidden, input_shape=self.input_shape, return_sequences=True))
            elif i == self.layers - 1:
                # last layer
                model.add(keras_lstm(self.hidden))
            else:
                # intermmediate layer
                model.add(keras_lstm(self.hidden, return_sequences=True))
            model.add(Dropout(0.3))

        model = DNN.add_softmax(self, model)

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        if self.config.is_debug():
            debug("Inputs: {}".format(model.inputs))
            model.summary()
            debug("Outputs: {}".format(model.outputs))
        return model
