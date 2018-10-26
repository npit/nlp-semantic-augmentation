import logging
import pickle
import os
import random

from sklearn import metrics
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.simplefilter(action='ignore', category=UndefinedMetricWarning)

import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Embedding, Reshape,Flatten
from keras.layers import LSTM as keras_lstm
from keras.utils import to_categorical
from keras import callbacks

from utils import info, debug, tic, toc, error


class DNN:
    save_dir = "models"
    folds = None
    performance = {}
    cw_performance = {}
    run_types = ['random', 'majority', 'run']
    measures = ["precision", "recall", "f1-score", "accuracy"]
    measure_aggregations = ["macro", "micro", "classwise", "weighted"]
    sequence_length = None


    do_train_embeddings = False
    train_embeddings_params = []

    def __init__(self, params):
        for x in self.run_types:
            self.performance[x] = {}
            for measure in self.measures:
                self.performance[x][measure] = {}
                for aggr in self.measure_aggregations:
                    self.performance[x][measure][aggr] = []

    # aggregated evaluation measure function shortcuts
    def get_pre_rec_f1(self, preds, metric, gt=None):
        if gt is None:
            gt = self.test_labels
        cr = pd.DataFrame.from_dict(metrics.classification_report(gt, preds, output_dict=True))
        # classwise, micro, macro, weighted
        cw = cr.loc[metric].iloc[:-3].as_matrix()
        mi = cr.loc[metric].iloc[-3]
        ma = cr.loc[metric].iloc[-2]
        we = cr.loc[metric].iloc[-1]
        return cw, mi, ma, we

    def acc(self, preds, gt=None):
        if gt is None:
            gt = self.test_labels
        return metrics.accuracy_score(gt, preds)


    def cw_acc(self, preds, gt=None):
        if gt is None:
            gt = self.test_labels
        cm = metrics.confusion_matrix(gt, preds)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        return cm.diagonal()


    def check_embedding_training(self, model):
        if self.do_train_embeddings:
            model.add(Embedding(self.vocabulary_size + 1, self.embedding_dim, input_length = self.sequence_length))
        return model

    # define useful keras callbacks for the training process
    def get_callbacks(self, config, fold_index="x"):
        self.callbacks = []
        results_folder = config.get_results_folder()
        if config.explicit_run_id():
            # if id is explicitly specified, use it
            run_name = config.get_run_id()
        else:
            # else use the autogenerated id and the learner name
            run_name = "{}_{}".format(self.name, config.get_run_id())

        models_folder = os.path.join(results_folder, run_name, "models")
        logs_folder = os.path.join(results_folder, run_name,"logs")
        [os.makedirs(x, exist_ok=True) for x in  [results_folder, models_folder, logs_folder]]
        self.results_folder = os.path.join(results_folder, run_name)

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

    def make(self, embedding, targets, num_labels, config):
        self.config = config
        if embedding.name == "train":
            info("Will train embeddings.")
            self.do_train_embeddings = True
            self.do_train_embeddings = True
            self.embedding_dim = embedding.get_dim()
            self.vocabulary_size = embedding.get_vocabulary_size()
            emb_seqlen = embedding.sequence_length
            if self.sequence_length is not None:
                if emb_seqlen != self.sequence_length:
                    error("Specified embedding sequence of length {}, but learner sequence is of length {}".format(emb_seqlen, self.sequence_length))
            self.sequence_length = emb_seqlen

        #import pdb; pdb.set_trace()
        self.verbosity = 1 if config.get_log_level() == "debug" else 0
        self.train, self.test = embedding.get_data()
        self.num_labels = num_labels
        self.train_labels, self.test_labels = [np.asarray(x, np.int32) for x in targets]
        self.input_dim = embedding.get_dim()

        train_params = config.get_train_params()
        self.epochs = train_params["epochs"]
        self.folds = train_params["folds"]
        self.early_stopping = train_params["early_stopping_patience"] if "early_stopping_patience" in train_params and train_params["early_stopping_patience"] > 0 else None
        self.seed = config.get_seed()
        self.batch_size = train_params["batch_size"]
        self.validation_portion = train_params["validation_portion"]

    def process_input(self, data):
        if self.do_train_embeddings:
            # reshape as per the sequence
            return np.reshape(data, (-1, self.sequence_length))
        return data

    def train_model(self, config):
        tic()
        model = self.get_model()
        train_y_onehot = to_categorical(self.train_labels, num_classes = self.num_labels)

        # shape accordingly
        self.train = self.process_input(self.train)
        history = model.fit(self.train, train_y_onehot,
                            batch_size=self.batch_size,
                            epochs=self.epochs,
                            verbose = self.verbosity,
                            validation_split= self.validation_portion,
                            callbacks = self.get_callbacks(config))
        if self.early_stopping:
            info("Stopped on epoch {}".format(self.early_stopping.stopped_epoch))
        self.do_test(model)
        toc("Total training")
        return model

    def do_traintest(self, config):
        if self.folds > 1:
            self.train_model_crossval(config)
        else:
            self.train_model(config)


    def train_model_crossval(self, config):
        tic()
        info("Training {} with input data: {} on {} stratified folds".format(self.name, len(self.train), self.folds))

        fold_data = self.get_fold_indexes()
        for fold_index, (train_d_idx, train_l_idx, val_d_idx, val_l_idx) in enumerate(fold_data):
            train_x, train_y = self.get_fold_data(self.train, self.train_labels, train_d_idx, train_l_idx)
            val_x, val_y = self.get_fold_data(self.train, self.train_labels, val_d_idx, val_l_idx)
            # convert labels to one-hot
            train_y_onehot = to_categorical(train_y, num_classes = self.num_labels)
            val_y_onehot = to_categorical(val_y, num_classes = self.num_labels)

            # train
            model = self.get_model()
            #print(val_x, val_y)
            info("Trainig fold {}/{}".format(fold_index + 1, self.folds))
            history = model.fit(train_x, train_y_onehot,
                                batch_size=self.batch_size,
                                epochs=self.epochs,
                                validation_data = (val_x, val_y_onehot),
                                verbose = self.verbosity,
                                callbacks = self.get_callbacks(config, fold_index))

            if self.early_stopping:
                info("Stopped on epoch {}".format(self.early_stopping.stopped_epoch))
            self.do_test(model)
        toc("Total training")
        # report results across folds
        self.report()

    # print performance across folds
    def report(self):
        info("==============================")
        info("Mean performance across all {} folds:".format(self.folds))
        res = {}
        for type in self.run_types:
            for measure in self.measures:
                for aggr in self.measure_aggregations:
                    if aggr not in self.performance[type][measure] or aggr == "classwise":
                        continue
                    container = self.performance[type][measure][aggr]
                    if not container:
                        continue
                    mean_perf = np.mean(container)
                    info("{:10} {:10} {:10} : {:.3f}".format(type, aggr, measure, mean_perf))
                    self.performance[type][measure][aggr].append(mean_perf)
        # write the results in csv in the results directory
        # entries in a run_type - measure configuration list are the foldwise scores, followed by the mean
        df = pd.DataFrame.from_dict(self.performance)
        df.to_csv(os.path.join(self.results_folder, "results.txt"))
        with open(os.path.join(self.results_folder, "results.pickle"), "wb") as f:
            pickle.dump(df, f)


    def do_test(self, model):
        if self.folds > 1:
            test_data, _ = self.get_fold_data(self.test)
        else:
            test_data = self.process_input(self.test)
        predictions = model.predict(test_data, batch_size=self.batch_size, verbose=self.verbosity)
        predictions_amax = np.argmax(predictions, axis=1)
        # get baseline performances
        self.compute_performance(predictions_amax)
        if self.config.is_debug():
            self.print_performance()

    # get fold data
    def get_fold_indexes_sequence(self):
        idxs = []
        skf = StratifiedKFold(self.folds, shuffle=False, random_state=self.seed)
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

    # fold generator function
    def get_fold_indexes(self):
        if len(self.train) != len(self.train_labels):
            return self.get_fold_indexes_sequence()
        else:
            skf = StratifiedKFold(self.folds, shuffle=False, random_state = self.seed)
            return [(train, train, val, val) for (train, val) in skf.split(self.train, self.train_labels)]

    # data preprocessing function
    def get_fold_data(self, data, labels=None, data_idx=None, label_idx=None):
        # if indexes provided, take only these parts
        if data_idx is not None:
            x = data[data_idx]
        else:
            x = data
        if labels is not None:
            if label_idx is not None:
                y = labels[label_idx]
            else:
                y = labels
        else:
            y = None
        return x, y

    # add softmax classification layer
    def add_softmax(self, model, is_first=False):
        if is_first:
            model.add(Dense(self.num_labels, input_shape=self.input_shape, name="dense_classifier"))
        else:
            model.add(Dense(self.num_labels, name="dense_classifier"))

        model.add(Activation('softmax', name="softmax"))
        return model

    # compute classification baselines
    def compute_performance(self, predictions):
        # add run performance
        self.add_performance("run", predictions)
        maxfreq, maxlabel = -1, -1
        for t in set(self.test_labels):
            freq = len([1 for x in self.test_labels if x == t])
            if freq > maxfreq:
                maxfreq = freq
                maxlabel = t

        majpred = np.repeat(maxlabel, len(self.test_labels))
        self.add_performance("majority", majpred)
        randpred = np.asarray([random.choice(list(range(self.num_labels))) for _ in self.test_labels], np.int32)
        self.add_performance("random", randpred)

    # compute scores and append to per-fold lists
    def add_performance(self, type, preds):
        # get accuracies
        acc, cw_acc = self.acc(preds), self.cw_acc(preds)
        self.performance[type]["accuracy"]["classwise"].append(cw_acc)
        self.performance[type]["accuracy"]["macro"].append(acc)
        # self.performance[type]["accuracy"]["micro"].append(np.nan)
        # self.performance[type]["accuracy"]["weighted"].append(np.nan)

        # get everything else
        for measure in [x for x in self.measures if x !="accuracy"]:
            cw, ma, mi, ws = self.get_pre_rec_f1(preds, measure)
            self.performance[type][measure]["classwise"].append(cw)
            self.performance[type][measure]["macro"].append(ma)
            self.performance[type][measure]["micro"].append(mi)
            self.performance[type][measure]["weighted"].append(ws)


    # print aggregate performance of the latest run
    def print_performance(self):
        info("---------------")
        for type in self.run_types:
            info("{} performance:".format(type))
            for measure in self.measures:
                for aggr in self.measure_aggregations:
                    if aggr not in self.performance[type][measure] or aggr == "classwise":
                        continue
                    container = self.performance[type][measure][aggr]
                    if not container:
                        continue
                    info('{:10} {:10}: {:.3f}'.format(aggr, measure, self.performance[type][measure][aggr][-1]))

class MLP(DNN):
    name = "mlp"
    def __init__(self, params):
        DNN.__init__(self, params)
        params = list(map(int, params))
        self.hidden = params[0]
        self.layers = params[1]

    def check_embedding_training(self, model):
        model = DNN.check_embedding_training(self, model)
        # vectorize
        model.summary()
        model.add(Reshape(target_shape=(-1, self.embedding_dim)))
        return model

    def make(self, embeddings, targets, num_labels, config):
        info("Building dnn: {}".format(self.name))
        DNN.make(self, embeddings, targets, num_labels, config)
        self.input_shape = (self.input_dim,)
        aggr = config.get_aggregation().split(",")
        aggregation = aggr[0]
        if aggregation not in ["avg"] and not self.do_train_embeddings:
            error("Aggregation {} incompatible with {} model.".format(aggregation, self.name))
        if embeddings.name == "train":
            error("{} cannot be used to train embeddings.".format(self.name))

    # build MLP model
    def get_model(self):
        model = Sequential()
        model = self.check_embedding_training(model)
        for i in range(self.layers):
            if i == 0 and not self.do_train_embeddings:
                model.add(Dense(self.hidden, input_shape=self.input_shape))
                model.add(Activation('relu'))
                model.add(Dropout(0.3))
            else:
                model.add(Dense(self.hidden))
                model.add(Activation('relu'))
                model.add(Dropout(0.3))

        model = DNN.add_softmax(self, model)
        model.summary()
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model


class LSTM(DNN):
    name = "lstm"

    def __init__(self, params):
        if len(params) < 2:
            error("Need lstm parameters: hidden size, sequence_length.")
        self.hidden = int(params[0])
        self.layers = int(params[1])
        self.sequence_length = int(params[2])
        DNN.__init__(self, params)

    # make network
    def make(self, embeddings, targets, num_labels, config):
        info("Building dnn: {}".format(self.name))
        DNN.make(self, embeddings, targets, num_labels, config)
        # make sure embedding aggregation is compatible
        # with the sequence-based lstm model
        self.input_shape = (self.sequence_length, self.input_dim)
        aggr = config.get_aggregation().split(",")
        aggregation = aggr[0]
        if aggregation not in ["pad"]:
            error("Aggregation {} incompatible with {} model.".format(aggregation, self.name))
        if config.get_embedding().startswith("train") in ["train"]:
            error("Embedding {} incompatible with {} model.".format(aggregation, self.name))



    # fetch sequence lstm fold data
    def get_fold_data(self, data, labels=None, data_idx=None, label_idx=None):
        # handle indexes by parent's function
        x, y = DNN.get_fold_data(self, data, labels, data_idx, label_idx)
        # reshape input data to num_docs x vec_dim x seq_len
        if not self.do_train_embeddings:
            x = np.reshape(x, (-1, self.sequence_length, self.input_dim))
        else:
            x = np.reshape(x, (-1, self.sequence_length))
            # replicate labels to match each input
            # y = np.stack([y for _ in range(self.sequence_length)])
            # y = np.reshape(np.transpose(y), (-1,1))
            pass


        return x, y

    # preprocess input
    def process_input(self, data):
        if self.do_train_embeddings:
            return DNN.process_input(self, data)
        return np.reshape(data, (-1, self.sequence_length, self.input_dim))

    # build the lstm model
    def get_model(self):
        model = Sequential()
        model = self.check_embedding_training(model)
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
        model.summary()

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        return model

