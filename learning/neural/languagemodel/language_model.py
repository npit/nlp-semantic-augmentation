"""Module for the incorporation of pretrained language models"""
from learning.neural.dnn import SupervisedDNN
import defs
from utils import error, info, one_hot
import numpy as np
from bundle.datatypes import *
from bundle.datausages import *


class NeuralLanguageModel(SupervisedDNN):
    """Class to implement a neural language model that ingests text sequences"""

    def __init__(self):
        """
        Constructor
        """
        super().__init__(self.config)

    def build_model(self):
        """In language models, the neural model is already built to generate embeddings from text"""
        pass

    def acquire_embedding_information(self):
        """Embedding acquisition for language models"""
        # produce input token embeddings from input text instead;
        # initialize the model
        info("Preparing the language model.")
        self.configure_language_model()

        # read necessary inputs
        self.fetch_language_model_inputs()

        # produce embeddings and/or masks
        info("Tokenizing LM textual input data to tokens")
        self.map_text()

    def fetch_language_model_inputs(self):
        # read input texts
        texts = self.data_pool.request_data(Text, Indices, usage_matching="subset", usage_exclude=GroundTruth, client=self.name)
        self.text = texts.data
        self.indices = texts.get_usage(Indices.name)

    # def encode_text(self, text):
    #     """Encode text into a sequence of tokens"""
    #     # return self.model.encode_text(text)
    #     return self.encode_text(text)

    def map_text(self):
        """Process input text into tokenized elements"""
        try:
            self.sequence_length = int(self.config.sequence_length)
        except ValueError:
            error(f"Need to set a sequence length for {self.name}")
        except TypeError:
            error(f"Need to set a sequence length for {self.name}")

        self.embeddings, self.masks, self.train_embedding_index, self.test_embedding_index = \
             self.map_text_collection(self.text, self.indices)

    def map_text_collection(self, texts, indices):
        """Encode a collection of texts into tokens, masks and train/test indexes"""
        train_index, test_index = [], []
        tokens, masks = [], []
        for i in range(len(texts.instances)):
            texts_instance = texts.instances[i]
            role = indices.roles[i]
            for doc_data in texts_instance:
                words = doc_data["words"]
                text = " ".join(words)
                toks, mask = self.encode_text(text)
                tokens.append(toks)
                masks.append(mask)

            idxs = list(range(len(texts_instance)))
            if role == defs.roles.train:
                train_index.extend([x + len(tokens) for x in idxs])
            elif role == defs.roles.test:
                test_index.extend([x + len(tokens) for x in idxs])

        tokens = np.concatenate(tokens)
        masks = np.concatenate(masks)
        train_index = np.asarray(train_index, dtype=np.long)
        test_index = np.asarray(test_index, dtype=np.long)
        return tokens, masks, train_index, test_index