"""Module for n-gram graph representations"""
import numpy as np
import tqdm

import defs
from pyinsect.collector.NGramGraphCollector import NGramGraphCollector
from representation.representation import Representation
from utils import (debug, error, info, realign_embedding_index, tictoc,
                   warning, write_pickled)


class NGG(Representation):
    """The n-gram graph representation class"""
    name = "ngg"

    def __init__(self, config):
        """The constructor for an n-gram graph converter
        """
        self.config = config
        self.base_name = self.name

        self.ngram_size = 3
        self.window_size = 3
        super().__init__()

    def process_component_inputs(self):
        """Additional checks for NGG"""
        super().process_component_inputs()
        # ensure labels are provided
        error(f"Component {self.base_name} requires labels in the input. ", not self.inputs.has_labels())
        error(f"Component {self.base_name} cannot be applied to multilabel data.", self.inputs.get_labels().multilabel)

    def map_text(self):
        """Map text to n-gram graph similarities"""
        if self.loaded_preprocessed or self.loaded_aggregated:
            return
        info("Mapping to {} word embeddings.".format(self.name))


        self.build_label_graphs()
        self.embeddings = self.build_label_vectors()

        self.set_identity_indexes(self.text)
        self.set_constant_elements_per_instance()

        info("Writing label graph embedding mappings to {}".format(self.serialization_path_preprocessed))
        write_pickled(self.serialization_path_preprocessed, self.get_all_preprocessed())

    def get_raw_text(self, instance):
        """Fetch raw text from a word list - POS tuple"""
        return " ".join([t[0] for t in instance])

    def build_label_graphs(self):
        """Build representative graphs for each class"""
        self.label_graphs = {}
        labelset = self.inputs.get_labels().labelset
        for l in labelset:
            self.label_graphs[l] = NGramGraphCollector()
        for texts, labels, role in zip(self.inputs.get_text().instances, self.inputs.get_labels().instances, self.inputs.get_indices().roles):
            # only use data marked for training
            if role != defs.roles.train:
                continue
            self.update_train(texts, labels)

    def update_train(self, texts, labels):
        """Update class graphs for a collection of texts"""
        with tqdm.tqdm(total=len(texts), ascii=True, desc="Updating label graphs.") as pbar:
            for text, label in zip(texts, labels):
                # get just text components
                text = self.get_raw_text(text)
                self.label_graphs[label.item()].addText(text, bDeepCopy=False, n=self.ngram_size, Dwin=self.window_size)
                pbar.update()

    def build_label_vectors(self):
        """Produce similarity vectors for each instance"""
        # prealloc all embeddings
        num_all_instances = sum(len(x) for x in self.text)
        embeddings = np.zeros((num_all_instances, len(self.label_graphs)), np.float32)
        # iterate over all texts
        with tqdm.tqdm(total=num_all_instances, desc="Encoding to graph similarities", ascii=True) as pbar:
            global_idx = 0
            for texts in self.inputs.get_text().instances:
                for text in texts:
                    text = self.get_raw_text(text)
                    # iterate over all class vectors
                    for l, label in enumerate(self.label_graphs):
                        sim_value = self.label_graphs[label].getAppropriateness(text, n=self.ngram_size, Dwin=self.window_size)
                        embeddings[global_idx][l] = sim_value
                    global_idx += 1
                    pbar.update()
        return embeddings

    def get_all_preprocessed(self):
        """Preprocessed data getter"""
        return {"embeddings": self.embeddings, "dataset_vectors": self.dataset_vectors, "elements_per_instance": self.elements_per_instance}
