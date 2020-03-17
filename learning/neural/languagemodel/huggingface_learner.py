"""Module for the incorporation of transformer models from huggingface"""
import numpy as np

import defs
import torch
from learning.neural.languagemodel.language_model import LanguageModel
from transformers import (BertConfig, BertForSequenceClassification, BertModel,
                          BertTokenizer)
from utils import error, info, one_hot


class HuggingfaceTransformer(LanguageModel):
    """Class to implement a huggingface transformer"""
    use_pretrained = True


    def __init__(self, config):
        """
        Keyword Arguments:
        config -- Configuration object
        """
        self.config = config
        self.requires_vector_info = False
        super().__init__()

    def map_text(self):
        """Process input text into tokenized elements"""
        try:
            self.sequence_length = int(self.config.sequence_length)
        except ValueError:
            error(f"Need to set a sequence length for {self.name}")
        except TypeError:
            error(f"Need to set a sequence length for {self.name}")

        self.embeddings = np.empty((0, self.sequence_length))
        self.test_embedding_index = np.ndarray((), np.int32)
        self.train_embedding_index = np.ndarray((), np.int32)
        text_instances = self.inputs.get_text().instances
        for i in range(len(self.inputs.get_indices().instances)):
            role = self.inputs.get_indices().roles[i]
            texts = self.inputs.get_text().instances[i]
            if role == defs.roles.train:
                self.train_embedding_index = np.append(self.train_embedding_index, np.arange(len(texts)))
            elif role == defs.roles.test:
                self.test_embedding_index = np.append(self.train_embedding_index, np.arange(len(texts)))

            # text has to be non-tokenized
            # input_ids = torch.tensor(tokenizer.encode(txt, add_special_tokens=True)).unsqueeze(0)  # Batch size 1
            texts = [" ".join([wpos[0] for wpos in doc]) for doc in texts]
            for txt in texts:
                ids = torch.tensor(self.tokenizer.encode(txt)).unsqueeze(0).numpy()
                # ids = torch.tensor(self.tokenizer.encode(txt, add_special_tokens=True).unsqueeze(0))
                sizediff = ids.shape[-1] - self.sequence_length
                if sizediff > 0:
                    ids = ids[:, :self.sequence_length]
                elif sizediff < 0:
                    ids = np.append(ids, np.zeros((1, -sizediff)), axis=1)
                self.embeddings = np.append(self.embeddings, ids, axis=0)

        self.train_index = np.arange(len(self.train_embedding_index))
        self.test_index = np.arange(len(self.test_embedding_index))
        info(f"{self.name} mapped text to input tokens")

    def train_model(self):
        train_data = self.get_data_from_index(self.train_index, self.embeddings)
        model = self.model(**self.args)
        model.fit(train_data, np.asarray(self.train_labels).ravel())
        return model

    # evaluate a clustering
    def test_model(self, model):
        model.eval()
        test_data = self.get_data_from_index(self.test_index, self.embeddings)
        outputs = self.model(test_data)
        loss, logits = outputs[:2]
        # convert back to one-hot
        predictions = one_hot(logits, self.num_labels)
        return predictions

    def make(self):
        pass


class Bert(HuggingfaceTransformer):
    name = "bert"

    def __init__(self, config):
        """
        Keyword Arguments:
        config -- config file
        """
        super().__init__(config)

    def make_language_model(self):
        """Assign model, tokenizer and pretrained weights"""
        self.num_labels = len(self.inputs.get_labels().labelset)
        pretrained_id = "bert-base-uncased"
        if self.use_pretrained:
            self.model = BertForSequenceClassification.from_pretrained(pretrained_id, num_labels=self.num_labels, output_hidden_states=False, output_attentions=False)
            self.tokenizer = BertTokenizer.from_pretrained(pretrained_id)
        else:
            self.model = BertForSequenceClassification(num_labels=self.num_labels,
                                                       output_hidden_states=False,
                                                       output_attentions=False)
            self.model = BertModel(num_labels=self.num_labels)
            self.tokenizer = BertTokenizer(pretrained_id, num_labels=self.num_labels)
