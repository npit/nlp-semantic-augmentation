from os.path import join, basename
import pandas as pd
from pandas.errors import ParserError
from utils import error, tictoc, info, debug, read_pickled, write_pickled, warning, shapes_list, read_lines, one_hot
import numpy as np
from serializable import Serializable
from semantic import SemanticResource
from bag import Bag, TFIDF
import defs
import copy


class Representation(Serializable):
    dir_name = "representation"
    loaded_transformed = False

    @staticmethod
    def create(config):
        name = config.representation.name
        if name == BagRepresentation.name:
            return BagRepresentation(config)
        if name == TFIDFRepresentation.name:
            return TFIDFRepresentation(config)
        # any unknown name is assumed to be pretrained embeddings
        return VectorEmbedding(config)

    @staticmethod
    def get_available():
        return [cls.name for cls in Representation.__subclasses__()]

    def __init__(self, can_fail_loading=True):
        self.set_params()
        self.set_name()
        Serializable.__init__(self, self.dir_name)
        # check for serialized mapped data
        self.set_serialization_params()
        # add paths for aggregated / transformed / enriched representations:
        # set required resources
        self.set_resources()
        # if a transform has been defined
        if self.config.has_transform():
            # suspend potential needless repr. loading for now
            return
        # fetch the required data
        self.acquire_data()

    # add exra representations-specific serialization paths
    def set_additional_serialization_sources(self):
        # compute names
        aggr = "".join(list(map(str, self.config.representation.aggregation + [self.sequence_length])))
        self.serialization_path_aggregated = "{}/{}.aggregated_{}.pickle".format(self.serialization_dir, self.name, aggr)

        sem = SemanticResource.generate_name(self.config, include_dataset=False)
        finalized_id = sem + "_" + self.config.semantic.enrichment if sem else "nosem"
        self.serialization_path_finalized = "{}/{}.aggregated_{}.finalized_{}.pickle".format(
            self.serialization_dir, self.name, aggr, finalized_id)

        # fill in at the desired order (finalized, transformed, aggregated
        self.data_paths = [self.serialization_path_finalized, self.serialization_path_aggregated] + self.data_paths
        self.read_functions = [read_pickled] * 2 + self.read_functions
        self.handler_functions = [self.handle_finalized, self.handle_aggregated] + self.handler_functions

    # shortcut for reading configuration values
    def set_params(self):
        self.dimension = self.config.representation.dimension
        self.dataset_name = self.config.dataset.name
        self.base_name = self.name

    @staticmethod
    def generate_name(config):
        return "{}_{}_dim{}".format(config.representation.name, config.dataset.full_name, config.representation.dimension)

    # name setter function, exists for potential overriding
    def set_name(self):
        self.name = Representation.generate_name(self.config)
        self.config.representation.full_name = self.name

    # finalize embeddings to use for training, aggregating all data to a single ndarray
    # if semantic enrichment is selected, do the infusion
    def set_semantic(self, semantic):
        if self.loaded_finalized:
            info("Skipping embeddings finalizing, since finalized data was already loaded.")
            return

        if self.config.semantic.enrichment is not None:
            if self.config.representation.name == "train":
                error("Semantic enrichment undefined for embedding training, for now.")
            semantic_data = semantic.get_vectors()
            info("Enriching [{}] embeddings with shapes: {} {} and {} vecs/doc with [{}] semantic information of shapes {} {}.".
                 format(self.config.representation.name, *shapes_list(self.dataset_vectors), self.vectors_per_doc, self.config.semantic.name, *shapes_list(semantic_data)))

            if self.config.semantic.enrichment == "concat":
                semantic_dim = len(semantic_data[0][0])
                final_dim = self.dimension + semantic_dim
                for dset_idx in range(len(semantic_data)):
                    info("Concatenating dataset part {}/{} to composite dimension: {}".format(dset_idx + 1, len(semantic_data), final_dim))
                    if self.vectors_per_doc > 1:
                        # tile the vector the needed times to the right, reshape to the correct dim
                        semantic_data[dset_idx] = np.reshape(np.tile(semantic_data[dset_idx], (1, self.vectors_per_doc)),
                                                             (-1, semantic_dim))
                    self.dataset_vectors[dset_idx] = np.concatenate(
                        [self.dataset_vectors[dset_idx], semantic_data[dset_idx]], axis=1)

            elif self.config.semantic.enrichment == "replace":
                final_dim = len(semantic_data[0][0])
                for dset_idx in range(len(semantic_data)):
                    info("Replacing dataset part {}/{} with semantic info of dimension: {}".format(dset_idx + 1, len(semantic_data), final_dim))
                    if self.vectors_per_doc > 1:
                        # tile the vector the needed times to the right, reshape to the correct dim
                        semantic_data[dset_idx] = np.reshape(np.tile(semantic_data[dset_idx], (1, self.vectors_per_doc)),
                                                             (-1, final_dim))
                    self.dataset_vectors[dset_idx] = semantic_data[dset_idx]
            else:
                error("Undefined semantic enrichment: {}".format(self.config.semantic.enrichment))

            # serialize finalized embeddings
            self.dimension = final_dim
            write_pickled(self.serialization_path_finalized, self.get_all_preprocessed())

    def handle_aggregated(self, data):
        self.handle_preprocessed(data)
        self.loaded_aggregated = True
        debug("Read aggregated dataset embeddings shapes: {}, {}".format(*shapes_list(self.dataset_vectors)))

    def handle_finalized(self, data):
        self.handle_preprocessed(data)
        self.loaded_finalized = True
        self.dimension = data["dataset_vectors"][0].shape[-1]
        debug("Read finalized dataset embeddings shapes: {}, {}".format(*shapes_list(self.dataset_vectors)))

    def get_zero_pad_element(self):
        return np.zeros((1, self.dimension), np.float32)

    def get_vocabulary_size(self):
        return len(self.dataset_words[0])

    def has_word(self, word):
        return word in self.embeddings.index

    def get_data(self):
        return self.dataset_vectors

    def get_dimension(self):
        return self.dimension

    # mark word-index relations for stat computation, add unk if needed
    def handle_raw_serialized(self, raw_serialized):
        """Read csv mapping in a pickled format
        """
        # process as dataframe
        self.words_to_numeric_idx = {}
        self.embeddings = raw_serialized
        for w in self.embeddings.index.tolist():
            self.words_to_numeric_idx[w] = len(self.words_to_numeric_idx)

    def handle_raw(self, raw_data):
        pass

    def fetch_raw(self, path):
        # assume embeddings are dataframes
        return None

    def preprocess(self):
        pass

    def loaded_enriched(self):
        return self.loaded_finalized

    def get_present_term_indexes(self):
        return self.present_term_indexes

    def get_vectors(self):
        return self.dataset_vectors

    def get_elements_per_instance(self):
        return self.elements_per_instance

    def get_min_elements_per_instance(self):
        return [min(x) for x in self.elements_per_instance]

    def match_targets_to_instances(self, dset_idx, gt, do_flatten=True, binarize_num_labels=None):
        """Expand, if needed, ground truth samples for multi-vector instances
        """
        epi = self.elements_per_instance[dset_idx]
        multi_vector_instance_idx = [i for i in range(len(epi)) if epi[i] > 1]
        if not multi_vector_instance_idx:
            if binarize_num_labels is not None:
                return one_hot(gt, num_labels=binarize_num_labels)
            return gt
        res = []
        for i in range(len(gt)):
            # get the number of elements for the instance
            times = epi[i]
            if do_flatten:
                res.extend([gt[i] for _ in range(times)])
            else:
                res.append([gt[i] for _ in range(times)])

        if binarize_num_labels is not None:
            return one_hot(res, num_labels=binarize_num_labels)
        return res

    def need_load_transform(self):
        return not (self.loaded_aggregated or self.loaded_finalized)


class Embedding(Representation):
    name = ""
    words = []
    dataset_vectors = None
    embeddings = None
    words_to_numeric_idx = None
    dimension = None
    sequence_length = None

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
        # word - vector correspondence
        try:
            self.embeddings = pd.read_csv(path, delimiter=" ", header=None, index_col=0)
        except ParserError as pe:
            error(pe.msg)

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
        if self.loaded_finalized:
            debug("Will not compute dense, since finalized data were loaded")
            return
        if self.loaded_transformed:
            debug("Will not compute dense, since transformed data were loaded")
            return

        info("Embeddings are already dense.")
        # instance vectors are already dense - just make dataset-level ndarrays
        for dset_idx in range(len(self.dataset_vectors)):
            self.dataset_vectors[dset_idx] = pd.concat(self.dataset_vectors[dset_idx]).values
            info("Computed dense shape for {}-sized dataset {}/{}: {}".format(len(self.dataset_vectors[dset_idx]), dset_idx + 1, len(self.dataset_vectors), self.dataset_vectors[dset_idx].shape))

    # prepare embedding data to be ready for classification
    def aggregate_instance_vectors(self):
        """Method that maps features to a single vector per instance"""
        if self.loaded_aggregated or self.loaded_finalized:
            debug("Skipping representation aggregation.")
            return
        info("Aggregating embeddings to single-vector-instances via the {} method.".format(self.aggregation))
        # use words per document for the aggregation, aggregating function as an argument
        # stats
        aggregation_stats = 0, 0

        for dset_idx in range(len(self.dataset_vectors)):
            info("Aggregating embedding vectors for collection {}/{} with shape {}".format(
                 dset_idx + 1, len(self.dataset_vectors), self.dataset_vectors[dset_idx].shape))

            aggregated_dataset_vectors = np.ndarray((0, self.dimension), np.float32)
            curr_idx = 0
            for inst_len in self.elements_per_instance[dset_idx]:
                curr_instance = self.dataset_vectors[dset_idx][curr_idx: curr_idx + inst_len]

                # average aggregation to a single vector
                if self.aggregation[0] == "avg":
                    curr_instance = np.mean(curr_instance, axis=0).reshape(1, self.dimension)
                # padding aggregation to specified vectors per instance
                elif self.aggregation[0] == "pad":
                    # filt = self.aggregation[1]
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
                else:
                    error("Undefined aggregation: {}".format(self.aggregation))

                aggregated_dataset_vectors = np.append(aggregated_dataset_vectors, curr_instance, axis=0)
                curr_idx += inst_len
            # update the dataset vector collection and dimension
            self.dataset_vectors[dset_idx] = aggregated_dataset_vectors
            # report stats
            if self.aggregation[0] == "pad":
                info("Truncated {:.3f}% and padded {:.3f} % items.".format(*[x / len(self.dataset_vectors[dset_idx]) * 100 for x in aggregation_stats]))

    # shortcut for reading configuration values
    def set_params(self):
        self.aggregation = self.config.representation.aggregation
        self.sequence_length = self.config.representation.sequence_length
        self.map_missing_unks = self.config.representation.missing_words == "unk"
        if type(self.aggregation) == list:
            if self.aggregation[0] == "pad":
                self.vectors_per_doc = self.sequence_length
            elif self.aggregation[0] == "avg":
                self.vectors_per_doc = 1
            else:
                error("Undefined aggregation: {}".format(self.aggregation))
        else:
            error("Undefined aggregation: {}".format(self.aggregation))
        Representation.set_params(self)


    def get_all_preprocessed(self):
        return {"dataset_vectors": self.dataset_vectors, "elements_per_instance": self.elements_per_instance,
                "undefined_word_index": None, "present_term_indexes": self.present_term_indexes}

    # mark preprocessing
    def handle_preprocessed(self, preprocessed):
        self.loaded_preprocessed = True
        self.dataset_vectors, self.elements_per_instance, \
        self.undefined_word_index, self.present_term_indexes = [preprocessed[n] for n in self.data_names]
        debug("Read preprocessed dataset embeddings shapes: {}, {}".format(*list(map(len, self.dataset_vectors))))

    def set_transform(self, transform):
        """Update representation information as per the input transform"""
        self.name += transform.get_name()
        self.dimension = transform.get_dimension()

        data = transform.get_all_preprocessed()
        self.dataset_vectors, self.elements_per_instance, self.undefined_word_index, \
            self.present_term_indexes = [data[n] for n in self.data_names]
        self.loaded_transformed = True


# generic class to load pickled embedding vectors
class VectorEmbedding(Embedding):
    name = "vector_embedding"
    unknown_word_token = "unk"

    # expected raw data path
    def get_raw_path(self):
        return "{}/{}_dim{}.pickle".format(self.raw_data_dir, self.base_name, self.dimension)

    # transform input texts to embeddings
    def map_text(self, dset):
        if self.loaded_preprocessed or self.loaded_aggregated or self.loaded_finalized:
            return
        info("Mapping dataset: {} to {} embeddings.".format(dset.name, self.name))
        text_bundles = dset.train, dset.test
        self.dataset_vectors = []
        self.present_term_indexes = []
        self.vocabulary = dset.vocabulary
        self.elements_per_instance = []

        if self.unknown_word_token not in self.embeddings and self.map_missing_unks:
            warning("[{}] unknown token missing from embeddings, adding it as zero vector.".format(self.unknown_word_token))
            self.embeddings.loc[self.unknown_word_token] = np.zeros(self.dimension)

        # loop over input text bundles (e.g. train & test)
        for dset_idx in range(len(text_bundles)):
            self.dataset_vectors.append([])
            self.present_term_indexes.append([])
            self.elements_per_instance.append([])
            with tictoc("Embedding mapping for text bundle {}/{}".format(dset_idx + 1, len(text_bundles))):
                info("Mapping text bundle {}/{}: {} texts".format(dset_idx + 1, len(text_bundles), len(text_bundles[dset_idx])))
                hist = {w: 0 for w in self.embeddings.index}
                hist_missing = {}
                num_documents = len(text_bundles[dset_idx])
                for j, doc_wp_list in enumerate(text_bundles[dset_idx]):
                    # drop POS
                    word_list = [wp[0] for wp in doc_wp_list]
                    debug("Text {}/{} with {} words".format(j + 1, num_documents, len(word_list)))
                    # check present & missing words
                    missing_words, missing_index, present_terms, present_index = [], [], [], []
                    for w, word in enumerate(word_list):
                        if word not in self.embeddings.index:
                            # debug("Word [{}] not in embedding index.".format(word))
                            missing_words.append(word)
                            missing_index.append(w)
                            if word not in hist_missing:
                                hist_missing[word] = 0
                            hist_missing[word] += 1
                        else:
                            present_terms.append(word)
                            present_index.append(w)
                            hist[word] += 1

                    # handle missing
                    if not self.map_missing_unks:
                        # ignore & discard missing words
                        word_list = present_terms
                    else:
                        # replace missing words with UNKs
                        for m in missing_index:
                            word_list[m] = self.unknown_word_token

                    if not present_terms and not self.map_missing_unks:
                        # no words present in the mapping, force
                        error("No words persent in document.")

                    # get embeddings
                    text_embeddings = self.embeddings.loc[word_list]
                    self.dataset_vectors[-1].append(text_embeddings)

                    # update present words and their index, per doc
                    self.elements_per_instance[-1].append(len(text_embeddings))
                    self.present_term_indexes[-1].append(present_index)

            self.print_word_stats(hist, hist_missing)

        # write
        info("Writing embedding mapping to {}".format(self.serialization_path_preprocessed))
        write_pickled(self.serialization_path_preprocessed, self.get_all_preprocessed())

    def print_word_stats(self, hist, hist_missing):
        terms_hit, hit_sum = sum([1 for v in hist if hist[v] > 0]), sum(hist.values())
        terms_missed, miss_sum = len([1 for v in hist_missing if hist_missing[v] > 0]), sum(hist_missing.values())
        total_term_sum = sum(list(hist.values()) + list(hist_missing.values()))
        debug("{} % terms appear at least once, which corresponds to a total of {} % terms in the text".format(terms_hit / len(hist) * 100, hit_sum / total_term_sum * 100))
        debug("{} % terms never appear, i.e. a total of {} % terms in the text".format(terms_missed / len(hist) * 100, miss_sum / total_term_sum * 100))

    def __init__(self, config):
        self.config = config
        self.name = self.base_name = self.config.representation.name
        Embedding.__init__(self)


class BagRepresentation(Representation):
    name = "bag"
    bag_class = Bag
    term_list = None
    do_limit = None

    data_names = ["dataset_vectors", "elements_per_instance", "term_list",
                  "present_term_indexes"]

    def __init__(self, config):
        self.config = config
        # check if a dimension meta file exists, to read dimension
        Representation.__init__(self)

    def set_params(self):
        self.vectors_per_doc = 1
        self.sequence_length = 1
        self.do_limit = False
        if self.config.representation.limit is not defs.limit.none:
            self.do_limit = True
            self.limit_type, self.limit_number = self.config.representation.limit

        Representation.set_params(self)

    @staticmethod
    def generate_name(config):
        name = Representation.generate_name(config)
        name_components = []
        if config.representation.limit is not defs.limit.none:
            name_components.append(defs.limit.to_string(config.representation.limit))
        return name + "_".join(name_components)

    def set_multiple_config_names(self):
        names = []
        # + no filtering, if filtered is specified
        filter_vals = [defs.limit.none]
        if self.do_limit:
            filter_vals.append(defs.limit.to_string(self.config.representation.limit))
        # any combo of weights, since they're all stored
        weight_vals = defs.weights.avail()
        for w in weight_vals:
            for f in filter_vals:
                conf = copy.copy(self.config)
                conf.representation.name = w
                conf.representation.limit = f
                candidate_name = self.generate_name(conf)
                names.append(candidate_name)
                debug("Bag config candidate: {}".format(candidate_name))
        self.multiple_config_names = names

    def set_name(self):
        # disable the dimension
        Representation.set_name(self)
        # if external term list, add its length to the name
        if self.config.representation.term_list is not None:
            self.name += "_tok_{}".format(basename(self.config.representation.term_list))
        self.name += defs.limit.to_string(self.config.representation.limit)

    def set_resources(self):
        if self.config.representation.term_list is not None:
            self.resource_paths.append(self.config.representation.term_list)
            self.resource_read_functions.append(read_lines)
            self.resource_handler_functions.append(self.handle_term_list)
            self.resource_always_load_flag.append(False)

    def handle_term_list(self, tok_list):
        info("Using external, {}-length term list.".format(len(tok_list)))
        self.term_list = tok_list

    def get_raw_path(self):
        return None

    def get_all_preprocessed(self):
        return {"dataset_vectors": self.dataset_vectors, "elements_per_instance": self.elements_per_instance,
                "term_list": self.term_list, "present_term_indexes": self.present_term_indexes}

    # sparse to dense
    def compute_dense(self):
        if self.loaded_finalized:
            debug("Will not compute dense, since finalized data were loaded")
            return
        if self.loaded_transformed:
            debug("Will not compute dense, since transformed data were loaded")
            return
        info("Computing dense representation for the bag.")
        self.dataset_vectors = Bag.generate_dense(self.dataset_vectors, self.term_list)
        info("Computed dense dataset shapes: {} {}".format(*shapes_list(self.dataset_vectors)))

    def aggregate_instance_vectors(self):
        # bag representations produce ready-to-use vectors
        pass

    def handle_preprocessed(self, preprocessed):
        self.loaded_preprocessed = True
        # intead of undefined word index, get the term list
        self.dataset_vectors, self.dataset_words, self.term_list, self.present_term_indexes = \
            [preprocessed[n] for n in self.data_names]
        # set misc required variables
        self.elements_per_instance = [[1 for _ in ds] for ds in self.dataset_vectors]

    def handle_aggregated(self, data):
        self.handle_preprocessed()
        self.loaded_aggregated = True
        # peek vector dimension
        data_dim = len(self.dataset_vectors[0][0])
        if self.dimension is not None:
            if self.dimension != data_dim:
                error("Configuration for {} set to dimension {} but read {} from data.".format(self.name, self.dimension, data_dim))
        self.dimension = data_dim

    def set_transform(self, transform):
        """Update representation information as per the input transform"""
        self.name += transform.get_name()
        self.dimension = transform.get_dimension()

        data = transform.get_all_preprocessed()
        self.dataset_vectors, self.elements_per_instance, self.term_list, self.present_term_indexes = \
            [data[n] for n in self.data_names]
        self.loaded_transformed = True

    def accomodate_dimension_change(self):
        self.set_name()
        self.set_serialization_params()
        self.set_additional_serialization_sources()

    def map_text(self, dset):
        if self.loaded_finalized or self.loaded_aggregated:
            debug("Skippping {} mapping due to preloading".format(self.base_name))
            return
        # need to calc term numeric index for aggregation
        if self.term_list is None:
            # no supplied token list -- use vocabulary of the training dataset
            self.term_list = dset.vocabulary
            info("Setting bag dimension to {} from dataset vocabulary.".format(len(self.term_list)))
        if self.do_limit:
            self.dimension = self.limit_number
        else:
            self.dimension = len(self.term_list)
        self.accomodate_dimension_change()
        info("Renamed representation after bag computation to: {}".format(self.name))

        # calc term index mapping
        self.term_index = {term: self.term_list.index(term) for term in self.term_list}

        # if self.dimension is not None and self.dimension != len(self.term_list):
        #     error("Specified an explicit bag dimension of {} but term list contains {} elements (delete it?).".format(self.dimension, len(self.term_list)))

        if self.loaded_preprocessed:
            debug("Skippping {} mapping due to preloading".format(self.base_name))
            return
        info("Mapping {} to {} representation.".format(dset.name, self.name))

        self.dataset_words = [self.term_list, None]
        self.dataset_vectors = []
        self.present_term_indexes = []

        # train
        self.bag = self.bag_class()
        self.bag.set_term_list(self.term_list)
        if self.do_limit:
            self.bag.set_term_filtering(self.limit_type, self.limit_number)
        self.bag.map_collection(dset.train)
        self.dataset_vectors.append(self.bag.get_weights())
        self.present_term_indexes.append(self.bag.get_present_term_indexes())
        if self.do_limit:
            self.term_list = self.bag.get_term_list()
            self.term_index = {k: v for (k, v) in self.term_index.items() if k in self.term_list}

        # # set representation dim and update name
        # self.dimension = len(self.term_list)
        # self.accomodate_dimension_change()

        # test
        self.bag = self.bag_class()
        self.bag.set_term_list(self.term_list)
        self.bag.map_collection(dset.test)
        self.dataset_vectors.append(self.bag.get_weights())
        self.present_term_indexes.append(self.bag.get_present_term_indexes())

        # set misc required variables
        self.elements_per_instance = [[1 for _ in ds] for ds in self.dataset_vectors]

        # write mapped data
        write_pickled(self.serialization_path_preprocessed, self.get_all_preprocessed())

        # if the representation length is not preset, write a small file with the dimension
        if self.config.representation.term_list is not None:
            with open(self.serialization_path_preprocessed + ".meta", "w") as f:
                f.write(str(self.dimension))


class TFIDFRepresentation(BagRepresentation):
    name = "tfidf"
    bag_class = TFIDF

    def __init__(self, config):
        BagRepresentation.__init__(self, config)

    # nothing to load, can be computed on the fly
    def fetch_raw(self, path):
        pass
