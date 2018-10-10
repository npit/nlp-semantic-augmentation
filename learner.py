import logging
import os
import numpy as np
import pickle
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout


class Dnn:
    name = "dnn"
    save_dir = "models"

    def __init__(self, params):
        pass

    def make(self, embeddings, targets, config):
        logger = logging.getLogger()
        logger.info("Building dnn")
        input_dim = embeddings[0][0].shape[-1]
        self.train, self.test = embeddings
        self.train_targets, self.test_targets = targets
        self.batch_size = config.get_batch_size()

        assert len(self.train_targets) == len(self.test_targets), "Train - test targets len mismatch."
        num_labels = len(self.train_targets)

        self.model = Sequential()
        self.model.add(Dense(512, input_shape=(input_dim,)))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(512))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(num_labels))
        self.model.add(Activation('softmax'))
        self.model.summary()

        self.model.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])

    def do_train(self):
        logger = logging.getLogger()
        logger.info("Training {}".format(self.name))
        history = self.model.fit(self.train, self.train_targets,
                            batch_size=self.batch_size,
                            epochs=30,
                            verbose=1,
                            validation_split=0.1)

    def do_test(self, text_labels):
        score = self.model.evaluate(self.test, self.test_targets,
                            batch_size=self.batch_size, verbose=1)

        print('Test accuracy:', score[1])


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
