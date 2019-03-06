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

    def get_current_model_path(self):
        filepath = join(self.models_folder, "{}".format(self.name))
        if self.do_folds:
            filepath += "_fold{}".format(self.fold_index)
        if self.do_validate_portion:
            filepath += "_valportion{}".format(self.validation_portion)
        return filepath

    # define useful keras callbacks for the training process
    def get_callbacks(self):
        self.callbacks = []
        [makedirs(x, exist_ok=True) for x in [self.results_folder, self.models_folder]]

        # model saving with early stoppingtch_si
        self.model_path = self.get_current_model_path()
        weights_path = self.model_path

        # weights_path = os.path.join(models_folder,"{}_fold_{}_".format(self.name, self.fold_index) + "ep_{epoch:02d}_valloss_{val_loss:.2f}.hdf5")
        self.model_saver = callbacks.ModelCheckpoint(weights_path, monitor='val_loss', verbose=0,
                                                     save_best_only=self.validation_exists, save_weights_only=False,
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

    # potentially apply DNN input data tranformations
    def process_input(self, data):
        return data

    # print information pertaining to early stopping
    def report_early_stopping(self):
        if self.validation_exists and self.early_stopping is not None:
            info("Stopped on epoch {}/{}".format(self.early_stopping.stopped_epoch + 1, self.epochs))
            write_pickled(self.model_path + ".early_stopping", self.early_stopping.stopped_epoch)

    # handle multi-vector items, expanding indexes to the specified sequence length
    def expand_index_to_sequence(self, fold_data):
        # map to indexes in the full-sequence data (e.g. times sequence_length)
        fold_data = list(map(lambda x: x * self.sequence_length if len(x) > 0 else np.empty((0,)), fold_data))
        for i in range(len(fold_data)):
            if fold_data[i] is None:
                continue
            # expand with respective sequence members (add an increment, vstack)
            stacked = np.vstack([fold_data[i] + incr for incr in range(self.sequence_length)])
            # reshape to a single vector, in the vertical (column) direction, that increases incrementally
            fold_data[i] = np.ndarray.flatten(stacked, order='F')
        return fold_data

    # split train/val labels and convert to one-hot
    def prepare_labels(self, trainval_idx):
        train_idx, val_idx = trainval_idx
        train_labels = self.train_labels
        if len(train_idx) > 0:
            train_labels = [self.train_labels[i] for i in train_idx]
            train_labels = one_hot(train_labels, self.num_labels)
        else:
            train_labels = np.empty((0,))
        if len(val_idx) > 0:
            val_labels = [self.train_labels[i] for i in val_idx]
            val_labels = one_hot(val_labels, self.num_labels)
        else:
            val_labels = np.empty((0,))
        return train_labels, val_labels

    # train a model on training & validation data portions
    def train_model(self, trainval_idx):
        # labels
        train_labels, val_labels = self.prepare_labels(trainval_idx)
        # data
        if self.num_train != self.num_train_labels:
            trainval_idx = self.expand_index_to_sequence(trainval_idx)
        train_data, val_data = [self.process_input(data) if len(data) > 0 else np.empty((0,)) for data in
                                [self.train[idx] if len(idx) > 0 else [] for idx in trainval_idx]]
        val_datalabels = (val_data, val_labels) if val_data.size > 0 else None
        # build model
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

    # add softmax classification layer
    def add_softmax(self, model, is_first=False):
        if is_first:
            model.add(Dense(self.num_labels, input_shape=self.input_shape, name="dense_classifier"))
        else:
            model.add(Dense(self.num_labels, name="dense_classifier"))

        model.add(Activation('softmax', name="softmax"))
        return model


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
        aggr = self.config.representation.aggregation
        aggregation = aggr[0]
        # sanity checks
        # sequence-based aggregation
        if aggregation not in ["pad"]:
            error("Aggregation [{}] incompatible with [{}] model.".format(aggregation, self.name))

        # non constant instance element num
        set_instance_lengths = [set(x) for x in representation.get_elements_per_instance()]
        if any([len(x) != 1 for x in set_instance_lengths]):
            error("[{}] needs a constant number of elements per instance per dataset, but got lengths: {}".format(self.name, ))
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
