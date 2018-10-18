import logging
import random
from sklearn import metrics
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.model_selection import StratifiedKFold

from utils import info, error, debug
import os
import numpy as np
import pickle
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from keras.layers import LSTM as keras_lstm
from keras.utils import to_categorical
from keras import callbacks

import warnings
warnings.simplefilter(action='ignore', category=UndefinedMetricWarning)


class DNN:
    save_dir = "models"
    folds = None
    performance = {}
    baseline = {}

    def __init__(self, params):
        for x in ['random', 'majority']:
            self.baseline[x] = {}
        for measure in ["acc", "ma_f1", "mi_f1"]:
            self.performance[measure] = []
            self.baseline['random'][measure] = []
            self.baseline['majority'][measure] = []

    # define useful keras callbacks for the training process
    def get_callbacks(self, config, fold_index):
        self.callbacks = []
        results_folder = config.get_results_folder()
        models_folder = os.path.join(results_folder, "run_{}_{}".format(self.name, config.get_run_id()), "models")
        logs_folder = os.path.join(results_folder, "run_{}_{}".format(self.name, config.get_run_id()),"logs")
        [os.makedirs(x, exist_ok=True) for x in  [results_folder, models_folder,logs_folder]]

        # model saving with early stopping
        model_path = os.path.join(models_folder,"{}_fold_{}".format(self.name, fold_index))
        self.model_saver =callbacks.ModelCheckpoint(model_path, monitor='val_loss', verbose=0,
                                                   save_best_only=True, save_weights_only=False,
                                                   mode='auto', period=1)
        self.callbacks.append(self.model_saver)
        if self.early_stopping:
            self.early_stopping = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=self.early_stopping, verbose=0,
                                                      mode='auto', baseline=None, restore_best_weights=False)
            self.callbacks.append(self.early_stopping)

        # stop on NaN
        self.nan_terminator = callbacks.TerminateOnNaN()
        self.callbacks.append(self.nan_terminator)
        # learning rate modifier at loss function plateaus
        self.lr_reducer = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                                          patience=10, verbose=0, mode='auto',
                                                          min_delta=0.0001, cooldown=0, min_lr=0)
        self.callbacks.append(self.lr_reducer)
        # logging
        log_file = os.path.join(logs_folder,"{}_fold_{}".format(self.name, fold_index))
        self.csv_logger = callbacks.CSVLogger(log_file, separator=',', append=False)
        self.callbacks.append(self.csv_logger)

        return self.callbacks

    def make(self, embeddings, targets, num_labels, config):

        self.verbosity = 1 if config.get_log_level() == "debug" else 0
        self.input_dim = embeddings[0][0].shape[-1]
        self.train, self.test = embeddings
        self.num_labels = num_labels
        self.train_labels, self.test_labels = targets
        self.batch_size = config.get_batch_size()

        train_params = config.get_train_params()
        self.epochs = train_params["epochs"]
        self.folds = train_params["folds"]
        self.early_stopping = train_params["early_stopping_patience"] if "early_stopping_patience" in train_params and train_params["early_stopping_patience"] > 0 else None
        self.seed = config.get_seed()

    def do_traintest(self, config):
        model = self.get_model()
        logger = logging.getLogger()
        logger.info("Training {} with input data {} on {} stratified folds".format(self.name, self.train.shape, self.folds))
        skf = StratifiedKFold(self.folds, shuffle=False, random_state = self.seed)
        fold_data = self.get_fold_indexes()
        for fold_index, (train_data_idx, train_label_idx, val_data_idx, val_label_idx) in enumerate(fold_data):
            train_x, train_y, val_x, val_y = self.get_fold_data(self.train, self.train_labels, train_data_idx, train_label_idx)
            val_x, val_y = self.get_fold_data(self.train, self.train_labels, val_data_idx, val_label_idx)
            # convert labels to one-hot
            train_y_onehot = to_categorical(train_y, num_classes = self.num_labels)
            val_y_onehot = to_categorical(val_y, num_classes = self.num_labels)

            # train
            logger.info("Trainig fold {}/{}".format(fold_index + 1, self.folds))
            history = model.fit(train_x, train_y_onehot,
                                batch_size=self.batch_size,
                                epochs=self.epochs,
                                verbose=self.verbosity,
                                validation_data = (val_x, val_y_onehot),
                                callbacks = self.get_callbacks(config, fold_index))

            if self.early_stopping:
                logger.info("Stopped on epoch {}".format(self.early_stopping.stopped_epoch))
            self.do_test(model)
        # report results across folds
        self.report()

    def report(self):
        logger = logging.getLogger()
        logger.info("==============================")
        logger.info("Mean performance across folds:")
        for measure, perfs in self.baseline['random'].items():
            logger.info("Random {} : {}".format(measure, np.mean(perfs)))
        for measure, perfs in self.baseline['majority'].items():
            logger.info("Majority {} : {}".format(measure, np.mean(perfs)))
        for measure, perfs in self.performance.items():
            logger.info("Run {} : {}".format(measure, np.mean(perfs)))

    def do_test(self, model):
        logger = logging.getLogger()
        #logger.info("Testing network.")
        test_data, = self.get_fold_data(self.test)
        predictions = model.predict(self.test, batch_size=self.batch_size, verbose=self.verbosity)
        predictions_amax = np.argmax(predictions, axis=1)
        self.get_baselines()
        acc   = metrics.accuracy_score(self.test_labels, predictions_amax)
        ma_f1 = metrics.f1_score(self.test_labels, predictions_amax, average='macro')
        mi_f1 = metrics.f1_score(self.test_labels, predictions_amax, average='micro')
        logger.info("---------------")
        logger.info("Run performance:")
        logger.info('Accuracy: {}'.format(acc))
        logger.info('Macro f1: {}'.format(ma_f1))
        logger.info('Micro f1: {}'.format(mi_f1))
        self.performance['acc'].append(acc)
        self.performance['mi_f1'].append(mi_f1)
        self.performance['ma_f1'].append(ma_f1)
        logger.info("Done testing network.")


    # fold generator function
    def get_fold_indexes(self):
        skf = StratifiedKFold(self.folds, shuffle=False, random_state = self.seed)
        return [(train, train, val, val) for (train, val) in enumerate(skf.split(self.train, self.train_labels))]

    # data preprocessing function
    def get_fold_data(self, data, labels, data_idx=None, label_idx=None):
        # if indexes provided, take only these parts
        if data_idx:
            x = data[data_idx]
        if label_idx and labels:
            y = labels[label_idx]
        elif labels:
            y = labels
        else:
            y = None
        return x, y

    def get_baselines(self):
        logger = logging.getLogger()
        logger.info("Baseline performance:")

        maxfreq, maxlabel = -1, -1
        for t in set(self.test_labels):
            freq = len([1 for x in self.test_labels if x == t])
            if freq > maxfreq:
                maxfreq = freq
                maxlabel = t

        majpred = np.repeat(maxlabel, len(self.test_labels))
        acc = metrics.accuracy_score(self.test_labels, majpred)
        ma_f1 = metrics.f1_score(self.test_labels, majpred, average='macro')
        mi_f1 = metrics.f1_score(self.test_labels, majpred, average='micro')
        logger.info("Majority classifier")
        logger.info('Accuracy: {}'.format(acc))
        logger.info('Macro f1: {}'.format(ma_f1))
        logger.info('Micro f1: {}'.format(mi_f1))
        self.baseline['majority']['acc'].append(acc)
        self.baseline['majority']['mi_f1'].append(mi_f1)
        self.baseline['majority']['ma_f1'].append(ma_f1)


        randpred = np.asarray([random.choice(list(range(self.num_labels))) for _ in self.test_labels], np.int32)
        acc = metrics.accuracy_score(self.test_labels, randpred)
        ma_f1 = metrics.f1_score(self.test_labels, randpred, average='macro')
        mi_f1 = metrics.f1_score(self.test_labels, randpred, average='micro')
        logger.info("Random classifier")
        logger.info('Accuracy: {}'.format(acc))
        logger.info('Macro f1: {}'.format(ma_f1))
        logger.info('Micro f1: {}'.format(mi_f1))
        self.baseline['random']['acc'].append(acc)
        self.baseline['random']['mi_f1'].append(mi_f1)
        self.baseline['random']['ma_f1'].append(ma_f1)

        # for i in range(10):
        #     prediction = self.model.predict(np.array([self.test[i]]))
        #     predicted_label = text_labels[np.argmax(prediction[0])]
        #     print(test_files_names.iloc[i])
        #     print('Actual label:' + test_tags.iloc[i])
        #     print("Predicted label: " + predicted_label)

    # def load(self):
    #     # load our saved model
    #     model = load_model('my_model.h5')

    #     # load tokenizer
    #     tokenizer = Tokenizer()
    #     with open('tokenizer.pickle', 'rb') as handle:
    #         tokenizer = pickle.load(handle)

    # def  save(self):
    #     # creates a HDF5 file 'my_model.h5'
    #     self.model.model.save(os.path.join(self.save_dir, 'my_model.h5'))

    #     # Save Tokenizer i.e. Vocabulary
    #     with open('tokenizer.pickle', 'wb') as handle:
    #         pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

class MLP(DNN):
    name = "mlp"
    def __init__(self, params):
        DNN.__init__(self, params)

    def make(self, embeddings, targets, num_labels, config):
        info("Building dnn: {}".format(self.name))
        DNN.make(self, embeddings, targets, num_labels, config)

    def get_model(self):
        model = Sequential()
        model.add(Dense(512, input_shape=(self.input_dim,)))
        model.add(Activation('relu'))
        model.add(Dropout(0.3))
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.3))
        model.add(Dense(self.num_labels))
        model.add(Activation('softmax'))
        model.summary()

        model.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])
        return model


class LSTM(DNN):
    name = "lstm"
    def __init__(self, params):
        if len(params) < 2:
            error("Need lstm parameters: hidden size, sequence_length.")
        self.sequence_length = int(params[1])
        self.hidden_length = int(params[0])
        DNN.__init__(self, params)


    def make(self, embeddings, targets, num_labels, config):
        info("Building dnn: {}".format(self.name))
        # make sure embedding aggregation is compatible with the sequence-based lstm model
        aggr = config.get_aggregation().split(",")
        aggregation = aggr[0]
        if aggregation not in ["pad"]:
            error("Aggregation {} incompatible with {} model.".format(aggregation, self.name))

        DNN.make(self, embeddings, targets, num_labels, config)

    # get fold data
    def get_fold_indexes(self):
        idxs = []
        skf = StratifiedKFold(self.folds, shuffle=False, random_state = self.seed)
        # get first-vector positions
        data_full_index = np.asarray((range(len(self.train))))
        single_vector_data = list(range(0, len(self.train), self.sequence_length))
        fold_data = list(skf.split(single_vector_data, self.train_labels))
        for train_test in fold_data:
            # get train indexes
            train_fold_singlevec_index, test_fold_singlevec_index = train_test
            # transform to full-sequence indexes
            train_fold_index = data_full_index[train_fold_singlevec_index]
            test_fold_index = data_full_index[test_fold_singlevec_index]
            # expand to the rest of the sequence members
            train_fold_index = [j for i in train_fold_index for j in list(range(i, i + self.sequence_length))]
            test_fold_index = [j for i in test_fold_index for j in list(range(i, i + self.sequence_length))]
            idxs.append((train_fold_index, train_fold_singlevec_index, test_fold_index, test_fold_singlevec_index))
        return idxs

    def get_fold_data(self, data, labels, data_idx=None, label_idx=None):
        # if indexes provided, take only these parts
        if data_idx:
            x = data[data_idx]
            # reshape input data
            import pdb; pdb.set_trace()
            x = np.reshape(x, (len(x), self.sequence_length, self.embedding_dim))
        if label_idx and labels:
            y = labels[label_idx]
        elif labels:
            y = labels
        else:
            y = None
        return x, y

    def get_model(self):
        model = Sequential()
        # model.add(Dense(512, input_shape=(self.input_dim,)))
        # model.add(Activation('relu'))
        # model.add(Dropout(0.3))
        model.add(keras_lstm(self.hidden_length, input_shape=(self.sequence_length, self.input_dim), time_steps = self.sequence_length))
        model.add(Dropout(0.3))
        model.add(Dense(self.num_labels))
        model.add(Activation('softmax'))
        model.summary()

        model.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])
        return model

