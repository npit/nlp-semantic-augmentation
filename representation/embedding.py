import defs
from representation.representation import Representation
from utils import error, info, debug, get_shape, shapes_list
import numpy as np
import pandas as pd
from pandas.errors import ParserError


class Embedding(Representation):
    name = ""
    words = []
    dataset_vectors = None
    embeddings = None
    words_to_numeric_idx = None
    dimension = None
    embedding_vocabulary_index = {}

    data_names = ["dataset_vectors", "elements_per_instance", "undefined_word_index",
                  "present_term_indexes"]

    def save_raw_embedding_weights(self, weights):
        error("{} is for pretrained embeddings only.".format(self.name))

    def set_resources(self):
        csv_mapping_name = "{}/{}.csv".format(self.raw_data_dir, self.base_name)
        self.resource_paths.append(csv_mapping_name)
        self.resource_read_functions.append(self.read_raw_embedding_mapping)
        self.resource_handler_functions.append(lambda x: x)

        # need the raw embeddings even if processed embedding data is available
        if self.config.has_semantic() and self.config.semantic.name == "context":
            # need the raw embeddings even if processed embedding data is available
            self.resource_always_load_flag.append(True)
            info("Forcing raw embeddings loading for semantic context embedding disambiguations.")

    def read_raw_embedding_mapping(self, path):
        # check if there's a vocabulary file
        try:
            raise FileNotFoundError
            warning("Partial map text is todo,")
            vocab_path = path + ".vocab"
            with open(vocab_path ) as f:
                lines = [x.strip() for x in f.readlines()]
                for word in [x for x in lines if x]:
                    self.embedding_vocabulary_index[word] = len(self.embedding_vocabulary_index)
            info("Read embedding vocabulary from path {}".format(vocab_path))
            return
        except FileNotFoundError:
            pass

        # word - vector correspondence
        try:
            self.embeddings = pd.read_csv(path, sep=self.config.misc.csv_separator, header=None, index_col=0)
        except ParserError as pe:
            error("Failed to read {}-delimited raw embedding from {}".format(self.config.misc.csv_separator, path), pe)
        # sanity check on defined dimension
        csv_dimension = self.embeddings.shape[-1]
        if csv_dimension != self.dimension:
            error("Specified representation dimension of {} but read csv embeddings are {}-dimensional.".format(self.dimension, csv_dimension))

    def __init__(self):
        Representation.__init__(self)

    # get vector representations of a list of words
    def get_embeddings(self, words):
        words = [w for w in words if w in self.embeddings.index]
        word_embeddings = self.embeddings.loc[words]
        # drop the nans and return
        return word_embeddings

    # for embeddings, vectors are already dense
    def get_dense_vector(self, vector):
        return vector

    # compute dense elements
    def compute_dense(self):
        pass
        # if self.loaded_finalized:
        #     debug("Will not compute dense, since finalized data were loaded")
        #     return
        # if self.loaded_transformed:
        #     debug("Will not compute dense, since transformed data were loaded")
        #     return

        # info("Embeddings are already dense.")
        # # instance vectors are already dense - just make dataset-level ndarrays
        # for dset_idx in range(len(self.dataset_vectors)):
        #     self.dataset_vectors[dset_idx] = pd.concat(self.dataset_vectors[dset_idx]).values
        #     info("Computed dense shape for {}-sized dataset {}/{}: {}".format(len(self.dataset_vectors[dset_idx]), dset_idx + 1, len(self.dataset_vectors), self.dataset_vectors[dset_idx].shape))

    # prepare embedding data to be ready for classification
    def aggregate_instance_vectors(self):
        """Method that maps features to a single vector per instance"""
        if self.aggregation == defs.alias.none:
            return
        if self.loaded_aggregated or self.loaded_finalized:
            debug("Skipping representation aggregation.")
            return
        info("Aggregating embeddings to single-vector-instances via the {} method.".format(self.aggregation))
        # use words per document for the aggregation, aggregating function as an argument
        # stats
        aggregation_stats = [0, 0]

        for dset_idx in range(len(self.dataset_vectors)):
            aggregated_dataset_vectors = np.ndarray((0, self.dimension), np.float32)
            info("Aggregating embedding vectors for collection {}/{} with shape {}".format(
                dset_idx + 1, len(self.dataset_vectors), get_shape(self.dataset_vectors[dset_idx])))

            new_numel_per_instance = []
            curr_idx = 0
            for inst_idx, inst_len in enumerate(self.elements_per_instance[dset_idx]):
                curr_instance = self.dataset_vectors[dset_idx][curr_idx: curr_idx + inst_len]
                if np.size(curr_instance) == 0:
                    error("Empty slice! Current index: {} / {}, instance numel: {}".format(curr_idx, len(self.dataset_vectors[dset_idx]), inst_len))

                # average aggregation to a single vector
                if self.aggregation == "avg":
                    curr_instance = np.mean(curr_instance, axis=0).reshape(1, self.dimension)
                    new_numel_per_instance.append(1)
                # padding aggregation to specified vectors per instance
                elif self.aggregation == "pad":
                    # filt = self.aggregation[1]
                    new_numel_per_instance.append(self.sequence_length)
                    num_vectors = len(curr_instance)
                    if self.sequence_length < num_vectors:
                        # truncate
                        curr_instance = curr_instance[:self.sequence_length, :]
                        aggregation_stats[0] += 1
                    elif self.sequence_length > num_vectors:
                        # make pad and stack vertically
                        pad_size = self.sequence_length - num_vectors
                        pad = np.tile(self.get_zero_pad_element(), (pad_size, 1), np.float32)
                        curr_instance = np.append(curr_instance, pad, axis=0)
                        aggregation_stats[1] += 1
                elif self.aggregation == defs.alias.none:
                    pass
                else:
                    error("Undefined aggregation: {}".format(self.aggregation))

                aggregated_dataset_vectors = np.append(aggregated_dataset_vectors, curr_instance, axis=0)
                curr_idx += inst_len
            # update the dataset vector collection and dimension
            self.dataset_vectors[dset_idx] = aggregated_dataset_vectors
            # update the elements per instance
            self.elements_per_instance[dset_idx] = new_numel_per_instance

            # report stats
            if self.aggregation == "pad":
                info("Truncated {:.3f}% and padded {:.3f} % items.".format(*[x / len(self.dataset_vectors[dset_idx]) * 100 for x in aggregation_stats]))
        info("Aggregated shapes: {}".format(shapes_list(self.dataset_vectors)))

    # shortcut for reading configuration values
    def set_params(self):
        self.map_missing_unks = self.config.representation.missing_words == "unk"
        # if self.aggregation == defs.aggregation.pad:
        #     pass
        # elif self.aggregation == defs.aggregation.avg:
        #     error("Sequence length of {} incompatible with {} aggregation".format(self.sequence_length, self.aggregation), \
        #           ill_defined(self.sequence_length, can_be=1))
        # elif self.aggregation == defs.alias.none:
        #     error("The {} representation requires an aggregation method.".format(self.base_name))
        # else:
        #     error("Undefined aggregation: {}".format(self.aggregation))
        # self.compatible_aggregations = defs.aggregation.avail + [defs.alias.none]
        self.compatible_aggregations = defs.aggregation.avail
        self.compatible_sequence_lengths = defs.sequence_length.avail
        Representation.set_params(self)

    def get_all_preprocessed(self):
        return {"dataset_vectors": self.dataset_vectors, "elements_per_instance": self.elements_per_instance,
                "undefined_word_index": None, "present_term_indexes": self.present_term_indexes}

    # mark preprocessing
    def handle_preprocessed(self, preprocessed):
        self.loaded_preprocessed = True
        self.dataset_vectors, self.elements_per_instance, \
        self.undefined_word_index, self.present_term_indexes = [preprocessed[n] for n in self.data_names]
        debug("Read preprocessed dataset embeddings shapes: {}".format(shapes_list(self.dataset_vectors)))
        #error("Read empty train or test preprocessed representations!", not all([x.size for x in self.dataset_vectors]))

    def set_transform(self, transform):
        """Update representation information as per the input transform"""
        self.name += transform.get_name()
        self.dimension = transform.get_dimension()

        data = transform.get_all_preprocessed()
        self.dataset_vectors, self.elements_per_instance, self.undefined_word_index, \
        self.present_term_indexes = [data[n] for n in self.data_names]
        self.loaded_transformed = True

