"""Module for the incorporation of pretrained language models"""
# from learning.neural.dnn import SupervisedDNN
import defs
from utils import error, info, one_hot
import numpy as np
from bundle.datatypes import *
from bundle.datausages import *
from tqdm import tqdm


class NLM:
    """Class to implement a neural language model
    The NLM ingests text sequence instead of numeric inputs"""

    def configure_language_model(self):
        """Do any preparatory actions specific to language models"""
        # override this
        pass

    def acquire_embedding_information(self):
        """Embedding acquisition function for language models
        This completely overrides the respective function which requests numeric data.
        """
        # produce input token embeddings from input text instead;
        # initialize the model
        info("Preparing the language model.")
        self.configure_language_model()

        # read necessary inputs
        self.fetch_language_model_inputs()

        # produce embeddings and/or masks
        self.map_text()

    def fetch_language_model_inputs(self):
        """Read data necessary for language model operation
        E.g. input texts, since LMs operate directly on text inputs
        """
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
        train_index, test_index = np.ndarray((0,), np.int32), np.ndarray((0,), np.int32)
        # plus 1 for the special cls token
        sql = self.sequence_length + 1
        tokens, masks = np.ndarray((0, sql), np.int64), np.ndarray((0,sql), np.int64)
        for i in range(len(indices.instances)):
            idx = indices.instances[i]
            # offset = len(tokens)
            for doc_idx in tqdm(idx, total=len(idx), desc=f"Encoding role: {indices.tags[i]}"):
                doc_data = texts.instances[doc_idx]
                words = doc_data["words"]
                text = " ".join(words)
                toks, mask = self.encode_text(text)
                tokens = np.append(tokens, toks, axis=0)
                masks = np.append(masks, mask, axis=0)

            # idxs = list(range(len(texts_instance)))
            # if role == defs.roles.train:
            #     train_index = np.append(train_idx, idx) extend([x + offset for x in idxs])
            # elif role == defs.roles.test:
            #     test_index.extend([x + offset for x in idxs])

        # tokens = np.concatenate(tokens)
        # masks = np.concatenate(masks)
        train_index, test_index = indices.get_train_test()
        # train_index = np.asarray(train_index, dtype=np.long)
        # test_index = np.asarray(test_index, dtype=np.long)
        return tokens, masks, train_index, test_index
