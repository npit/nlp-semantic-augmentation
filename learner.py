import logging
import random
from sklearn import metrics
from sklearn.exceptions import UndefinedMetricWarning
import os
import numpy as np
import pickle
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from keras.utils import to_categorical

import warnings
warnings.simplefilter(action='ignore', category=UndefinedMetricWarning)


class Dnn:
    name = "dnn"
    save_dir = "models"

    def __init__(self, params):
        pass

    def make(self, embeddings, targets, num_labels, config):
        logger = logging.getLogger()
        logger.info("Building dnn")
        input_dim = embeddings[0][0].shape[-1]
        self.train, self.test = embeddings
        self.num_labels = num_labels
        # convert to one-hot and ndarrays
        targets = [[to_categorical(y, num_classes=self.num_labels) for y in dset_targets] for dset_targets in targets]
        targets = [np.concatenate(t) for t in targets]
        self.train_targets, self.test_targets = [t.reshape(llen, num_labels) for t, llen in zip(targets, list(map(len,[self.train, self.test])))]
        #self.train_targets = np.concatenate().reshape(len(self.train_targets), self.num_labels)
        #self.test_targets = np.concatenate([to_categorical(y, num_classes=self.num_labels) for y in targets[-1]]).reshape(len(self.test_targets), self.num_labels)
        self.batch_size = config.get_batch_size()

        # assert len(self.train_targets) == len(self.test_targets), "Train - test targets len mismatch."

        self.model = Sequential()
        self.model.add(Dense(512, input_shape=(input_dim,)))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(512))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(self.num_labels))
        self.model.add(Activation('softmax'))
        self.model.summary()

        self.model.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])

        self.train_params = config.get_train_params()
        self.epochs = self.train_params["epochs"]

    def do_train(self):
        logger = logging.getLogger()
        logger.info("Training {} with input data {}".format(self.name, self.train.shape))
        history = self.model.fit(self.train, self.train_targets,
                            batch_size=5,
                            epochs=self.epochs,
                            verbose=1,
                            validation_split=0.1)

    def do_test(self):
        logger = logging.getLogger()
        logger.info("Testing network.")
        predictions = self.model.predict(self.test, batch_size=self.batch_size, verbose=1)
        predictions_amax, true_amax = np.argmax(predictions, axis=1), np.argmax(self.test_targets, axis=1)
        self.get_baselines()
        logger.info("---------------")
        logger.info("Run performance:")
        logger.info('Accuracy: {}'.format(metrics.accuracy_score(true_amax, predictions_amax)))
        logger.info('Macro f1: {}'.format(metrics.f1_score(true_amax, predictions_amax, average='macro')))
        logger.info('Micro f1: {}'.format(metrics.f1_score(true_amax, predictions_amax, average='micro')))

    def get_baselines(self):
        print()
        logger = logging.getLogger()
        logger.info("Baseline performance:")
        true_amax = np.argmax(self.test_targets, axis=1).tolist()

        maxfreq, maxlabel = -1, -1
        for t in set(true_amax):
            freq = len([1 for x in true_amax if x == t])
            if freq > maxfreq:
                maxfreq = freq
                maxlabel = t

        majpred = np.repeat(maxlabel, len(true_amax))
        logger.info("Majority classifier")
        logger.info('Accuracy: {}'.format(metrics.accuracy_score(true_amax, majpred)))
        logger.info('Macro f1: {}'.format(metrics.f1_score(true_amax, majpred, average='macro')))
        logger.info('Micro f1: {}'.format(metrics.f1_score(true_amax, majpred, average='micro')))


        randpred = np.asarray([random.choice(list(range(self.num_labels))) for _ in true_amax], np.int32)
        logger.info("Random classifier")
        logger.info('Accuracy: {}'.format(metrics.accuracy_score(true_amax, randpred)))
        logger.info('Macro f1: {}'.format(metrics.f1_score(true_amax, randpred, average='macro')))
        logger.info('Micro f1: {}'.format(metrics.f1_score(true_amax, randpred, average='micro')))

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
