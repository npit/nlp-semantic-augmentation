import pickle
from os.path import join, dirname
from os import makedirs
import random

from sklearn import metrics
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
import warnings
warnings.simplefilter(action='ignore', category=UndefinedMetricWarning)

import numpy as np
import pandas as pd

from keras.models import Sequential, load_model
from keras.layers import Activation, Dense, Dropout, Embedding, Reshape
from keras.layers import LSTM as keras_lstm
from keras.utils import to_categorical
from keras import callbacks

from utils import info, debug, tictoc, error, write_pickled

class DNN:
    save_dir = "models"
    folds = None
    performance = {}
    cw_performance = {}
    run_types = ['random', 'majority', 'run']
    measures = ["precision", "recall", "f1-score", "accuracy"]
    classwise_aggregations = ["macro", "micro", "classwise", "weighted"]
    stats = ["mean", "var", "std", "folds"]
    sequence_length = None

    do_train_embeddings = False
    train_embeddings_params = []
    do_folds = False

    model_paths = []

    def create(config):
        name = config.learner.name
        if name == LSTM.name:
            return LSTM(config)
        elif name == MLP.name:
            return MLP(config)
        else:
            error("Undefined learner: {}".format(name))

    def __init__(self):
        self.configure_evaluation_measures()
        pass

    # initialize evaluation containers and preferred evaluation printage
    def configure_evaluation_measures(self):
        info("Creating learner: {}".format(self.config.learner.to_str()))
        for run_type in self.run_types:
            self.performance[run_type] = {}
            for measure in self.measures:
                self.performance[run_type][measure] = {}
                for aggr in self.classwise_aggregations:
                    self.performance[run_type][measure][aggr] = {}
                    for stat in self.stats:
                        self.performance[run_type][measure][aggr][stat] = None
                    self.performance[run_type][measure][aggr]["folds"] = []
            # remove undefined combos
            for aggr in [x for x in self.classwise_aggregations if x not in ["macro", "classwise"]]:
                del self.performance[run_type]["accuracy"][aggr]

        # pritn only these, from config
        self.preferred_types = self.config.print.run_types if self.config.print.run_types else self.run_types
        self.preferred_measures = self.config.print.measures if self.config.print.measures else self.measures
        self.preferred_aggregations = self.config.print.aggregations if self.config.print.aggregations else self.classwise_aggregations
        self.preferred_stats = self.stats


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

    # get average accuracy
    def compute_accuracy(self, preds, gt=None):
        if gt is None:
            gt = self.test_labels
        return metrics.accuracy_score(gt, preds)


    # get class-wise accuracies
    def compute_classwise_accuracy(self, preds, gt=None):
        if gt is None:
            gt = self.test_labels
        cm = metrics.confusion_matrix(gt, preds)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        return cm.diagonal()


    # add an embedding layer, if necessary
    def check_add_embedding_layer(self, model):
        if self.do_train_embeddings:
            model.add(Embedding(self.vocabulary_size + 1, self.embedding_dim, input_length = self.sequence_length))
        return model

    # define useful keras callbacks for the training process
    def get_callbacks(self):
        self.callbacks = []
        self.results_folder = self.config.folders.results
        models_folder = join(self.results_folder, "models")
        logs_folder = self.results_folder
        [makedirs(x, exist_ok=True) for x in  [self.results_folder, models_folder, logs_folder]]

        # model saving with early stoppingtch_si
        self.model_path = join(models_folder,"{}_fold_{}_".format(self.name, self.fold_index))
        weights_path = self.model_path

        # weights_path = os.path.join(models_folder,"{}_fold_{}_".format(self.name, self.fold_index) + "ep_{epoch:02d}_valloss_{val_loss:.2f}.hdf5")
        self.model_saver = callbacks.ModelCheckpoint(weights_path, monitor='val_loss', verbose=0,
                                                   save_best_only=True, save_weights_only=False,
                                                   mode='auto', period=1)
        self.callbacks.append(self.model_saver)
        if self.early_stopping_patience:
            self.early_stopping = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=self.early_stopping_patience, verbose=0,
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
        log_file = join(logs_folder,"{}_fold_{}.csv".format(self.name, self.fold_index))
        self.csv_logger = callbacks.CSVLogger(log_file, separator=',', append=False)
        self.callbacks.append(self.csv_logger)
        return self.callbacks

    # to preliminary work
    def make(self, embeddings, targets, num_labels):
        if embeddings.base_name == "train":
            self.do_train_embeddings = True
            self.embedding_name = embeddings.name
            self.embedding_dim = embeddings.get_dim()
            info("Will train {}-dimensional embeddings.".format(self.embedding_dim))
            self.final_dim = embeddings.get_final_dim()
            self.vocabulary_size = embeddings.get_vocabulary_size()
            emb_seqlen = embeddings.sequence_length
            if self.sequence_length is not None:
                if emb_seqlen != self.sequence_length:
                    error("Specified embedding sequence of length {}, but learner sequence is of length {}".format(emb_seqlen, self.sequence_length))
            self.sequence_length = emb_seqlen
            self.embeddings = embeddings

        self.verbosity = 1 if self.config.log_level == "debug" else 0
        self.train, self.test = embeddings.get_data()
        self.num_labels = num_labels
        self.train_labels, self.test_labels = [np.asarray(x, np.int32) for x in targets]
        self.input_dim = embeddings.get_final_dim()

        self.epochs = self.config.train.epochs
        self.folds = self.config.train.folds
        self.do_folds = self.folds and self.folds > 1
        self.validation_portion = self.config.train.validation_portion
        self.early_stopping_patience = self.config.train.early_stopping_patience
        self.seed = self.config.get_seed()
        self.batch_size = self.config.train.batch_size

        if self.do_folds and self.validation_portion:
            error("Specified both folds {} and validation portion {}.".format(self.folds, self.validation_portion))

    # potentially apply DNN input data tranformations
    def process_input(self, data):
        if self.do_train_embeddings:
            # reshape as per the sequence
            return np.reshape(data, (-1, self.sequence_length))
        return data

    # print information pertaining to early stopping
    def report_early_stopping(self):
        if self.early_stopping_patience:
            info("Stopped on epoch {}/{}".format(self.early_stopping.stopped_epoch+1, self.epochs))
            write_pickled(self.model_path + ".early_stopping", self.early_stopping.stopped_epoch)


    # perfrom a train-test loop
    def do_traintest(self):
        # get trainval data
        train_val_idxs = self.get_trainval_indexes()

        # keep track of models' test performances and paths wrt selected metrics
        model_paths = []

        with tictoc("Total training", do_print=self.do_folds, announce=False):
            # loop on folds, or do a single loop on the train-val portion split
            for fold_index, trainval_idx in enumerate(train_val_idxs):
                self.fold_index = fold_index
                self.current_run_descr = "fold {}/{}".format(fold_index + 1, self.folds) if self.do_folds else \
                    "{}-val split".format(self.validation_portion)
                # train the model
                with tictoc("Training run {} on data :{}.".format(self.current_run_descr, list(map(len, trainval_idx)))):
                    model = self.train_model2(trainval_idx)
                # test the model
                with tictoc("Testing {} on data: {}.".format(self.current_run_descr, len(self.test_labels))):
                    self.do_test(model)
                    model_paths.append(self.model_saver.filepath)
            if self.do_folds:
                self.report_across_folds()
            # for embedding training, write the embeddings
            if self.do_train_embeddings:
                if self.do_folds:
                    # decide on best model wrt to first preferred, else macro f1
                    measure, aggr = [x[0] for x in [self.preferred_measures, self.preferred_aggregations]]
                    best_fold = np.argmax(self.performance['run'][measure][aggr][0])
                    model = load_model(model_paths[best_fold])
                else:
                    model = load_model(model_paths[0])
                # get the embedding weights
                weights = model.layers[0].get_weights()[0]
                self.embeddings.save_raw_embedding_weights(weights, dirname(self.model_path))
                pass



    # handle multi-vector items, expanding indexes to the specified sequence length
    def expand_index_to_sequence(self, fold_data):
        # map to indexes in the full-sequence data (e.g. times sequence_length)
        fold_data = list(map( lambda x: x * self.sequence_length, fold_data))
        for i in range(len(fold_data)):
            # expand with respective sequence members (add an increment, vstack)
            fold_data[i] = np.vstack([fold_data[i]+incr for incr in range(self.sequence_length)])
        return fold_data

    # train a model on training & validation data portions
    def train_model2(self, trainval_idx):
        # labels
        train_labels, val_labels = [to_categorical(data, num_classes=self.num_labels) for data in \
                                    [self.train_labels[idx] for idx in trainval_idx]]
        # data
        if len(self.train) != len(self.train_labels):
            trainval_idx = self.expand_index_to_sequence(trainval_idx)
        train_data, val_data = [self.process_input(data) for data in \
                                [self.train[idx] for idx in trainval_idx]]
        # build model
        model = self.get_model()
        # train the damn thing!
        model.fit(train_data, train_labels,
                    batch_size=self.batch_size,
                    epochs=self.epochs,
                    validation_data = (val_data, val_labels),
                    verbose = self.verbosity,
                    callbacks = self.get_callbacks())
        self.report_early_stopping()
        return model


    # print performance across folds and compute foldwise aggregations
    def report_across_folds(self):
        info("==============================")
        info("Mean / var / std performance across all {} folds:".format(self.folds))
        for type in self.run_types:
            for measure in self.measures:
                for aggr in self.classwise_aggregations:
                    if aggr not in self.performance[type][measure] or aggr == "classwise":
                        continue
                    container = self.performance[type][measure][aggr]
                    if not container:
                        continue
                    #print(type, measure, aggr, container)
                    mean_perf = np.mean(container["folds"])
                    var_perf = np.var(container["folds"])
                    std_perf = np.std(container["folds"])
                    # print, if it's prefered
                    if all([ type in self.preferred_types, measure in self.preferred_measures, aggr in self.preferred_aggregations]):
                        info("{:10} {:10} {:10} : {:.3f} {:.3f} {:.3f}".format(type, aggr, measure, mean_perf, var_perf, std_perf))
                    # add fold-aggregating performance information
                    self.performance[type][measure][aggr]["mean"] = mean_perf
                    self.performance[type][measure][aggr]["var"] = var_perf
                    self.performance[type][measure][aggr]["std"] = std_perf
        # write the results in csv in the results directory
        # entries in a run_type - measure configuration list are the foldwise scores, followed by the mean
        df = pd.DataFrame.from_dict(self.performance)
        df.to_csv(join(self.results_folder, "results.txt"))
        with open(join(self.results_folder, "results.pickle"), "wb") as f:
            pickle.dump(df, f)


    # evaluate a model on the test set
    def do_test(self, model):
        print_results = self.do_folds and self.config.print.folds or not self.folds
        test_data = self.process_input(self.test)
        predictions = model.predict(test_data, batch_size=self.batch_size, verbose=self.verbosity)
        predictions_amax = np.argmax(predictions, axis=1)
        # get baseline performances
        self.compute_performance(predictions_amax)
        if print_results:
            self.print_performance()

    # produce training / validation splits, with respect to sample indexes
    def get_trainval_indexes(self):
        if self.do_folds:
            info("Training {} with input data: {} on {} stratified folds".format(self.name, len(self.train), self.folds))
            splitter = StratifiedKFold(self.folds, shuffle=False, random_state = self.seed)
        else:
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=self.validation_portion)
        return list(splitter.split(np.zeros(len(self.train_labels)), self.train_labels))


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
        acc, cw_acc = self.compute_accuracy(preds), self.compute_classwise_accuracy(preds)
        self.performance[type]["accuracy"]["classwise"]["folds"].append(cw_acc)
        self.performance[type]["accuracy"]["macro"]["folds"].append(acc)
        # self.performance[type]["accuracy"]["micro"].append(np.nan)
        # self.performance[type]["accuracy"]["weighted"].append(np.nan)

        # get everything else
        for measure in [x for x in self.measures if x !="accuracy"]:
            cw, ma, mi, ws = self.get_pre_rec_f1(preds, measure)
            self.performance[type][measure]["classwise"]["folds"].append(cw)
            self.performance[type][measure]["macro"]["folds"].append(ma)
            self.performance[type][measure]["micro"]["folds"].append(mi)
            self.performance[type][measure]["weighted"]["folds"].append(ws)

    # print performance of the latest run
    def print_performance(self):
        info("---------------")
        info("Test results for {}:".format(self.current_run_descr))
        for type in self.preferred_types:
            info("{} performance:".format(type))
            for measure in self.preferred_measures:
                for aggr in self.preferred_aggregations:
                    # don't print classwise results or unedfined aggregations
                    if aggr not in self.performance[type][measure] or aggr == "classwise":
                        continue
                    container = self.performance[type][measure][aggr]
                    if not container:
                        continue
                    info('{} {}: {:.3f}'.format(aggr, measure, self.performance[type][measure][aggr]["folds"][self.fold_index]))
        info("---------------")

class MLP(DNN):
    name = "mlp"
    def __init__(self, config):
        self.config = config
        self.hidden = self.config.learner.hidden_dim
        self.layers = self.config.learner.num_layers
        self.sequence_length = self.config.learner.sequence_length
        DNN.__init__(self)


    def check_add_embedding_layer(self, model):
        if self.do_train_embeddings:
            error("Embedding training unsupported for {}".format(self.name))
            model = DNN.check_add_embedding_layer(self, model)
            # vectorize
            model.add(Reshape(target_shape=(-1, self.embedding_dim)))
        return model

    def make(self, embeddings, targets, num_labels):
        info("Building dnn: {}".format(self.name))
        DNN.make(self, embeddings, targets, num_labels)
        self.input_shape = (self.input_dim,)
        aggr = self.config.embedding.aggregation
        aggregation = aggr[0]
        if aggregation not in ["avg"] and not self.do_train_embeddings:
            error("Aggregation {} incompatible with {} model.".format(aggregation, self.name))
        if embeddings.name == "train":
            error("{} cannot be used to train embeddings.".format(self.name))

    # build MLP model
    def get_model(self):
        model = None
        model = Sequential()
        model = self.check_add_embedding_layer(model)
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
    def make(self, embeddings, targets, num_labels):
        info("Building dnn: {}".format(self.name))
        DNN.make(self, embeddings, targets, num_labels)
        # make sure embedding aggregation is compatible
        # with the sequence-based lstm model
        self.input_shape = (self.sequence_length, self.input_dim)
        aggr = self.config.embedding.aggregation
        aggregation = aggr[0]
        if aggregation not in ["pad"]:
            error("Aggregation {} incompatible with {} model.".format(aggregation, self.name))
        if aggr in ["train"]:
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
        model = self.check_add_embedding_layer(model)
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

