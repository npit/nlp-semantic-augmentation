"""Module for the incorporation of pretrained language models"""
from learning.neural.dnn import SupervisedDNN
import defs
from utils import error, info, one_hot
import numpy as np


class SupervisedNeuralLanguageModel(SupervisedDNN):
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
        self.configure_language_model(self.inputs.get_labels().labelset)
        # check existence
        error(f"{self.name} requires input texts", not self.inputs.has_text())
        # produce embeddings and/or masks
        self.map_text()

    def encode_text(self, text):
        """Encode text into a sequence of tokens"""
        # return self.model.encode_text(text)
        return self.encode_text(text)

    def map_text(self):
        """Process input text into tokenized elements"""
        try:
            self.sequence_length = int(self.config.sequence_length)
        except ValueError:
            error(f"Need to set a sequence length for {self.name}")
        except TypeError:
            error(f"Need to set a sequence length for {self.name}")

        self.embeddings = np.empty((0, self.sequence_length))
        self.masks = np.empty((0, self.sequence_length))

        self.test_embedding_index = np.ndarray((0,), np.int32)
        self.train_embedding_index = np.ndarray((0,), np.int32)

        texts = self.inputs.get_text()
        for i in range(len(texts.instances)):
            role = texts.roles[i]
            text = texts.instances[i]
            info(f"Feeding {role} text to the language model for tokenization.")
            if role == defs.roles.train:
                self.train_embedding_index = np.append(self.train_embedding_index, np.arange(len(text)))
            elif role == defs.roles.test:
                self.test_embedding_index = np.append(self.test_embedding_index, np.arange(len(text)))

            # text has to be non-tokenized
            # input_ids = torch.tensor(tokenizer.encode(txt, add_special_tokens=True)).unsqueeze(0)  # Batch size 1
            text = [" ".join([wpos[0] for wpos in doc]) for doc in text]
            for txt in text:
                ids, mask = self.encode_text(txt)
                # # ids = torch.tensor(self.tokenizer.encode(txt, add_special_tokens=True).unsqueeze(0))
                # sizediff = ids.shape[-1] - self.sequence_length
                # if sizediff > 0:
                #     ids = ids[:, :self.sequence_length]
                # elif sizediff < 0:
                #     ids = np.append(ids, np.zeros((1, -sizediff)), axis=1)
                self.embeddings = np.append(self.embeddings, ids, axis=0)
                self.masks = np.append(self.masks, mask, axis=0)

        info(f"{self.name} mapped text to input tokens")