from os.path import join, dirname
import pandas as pd
from dataset import Dataset
from utils import error, tictoc, info, debug, read_pickled, write_pickled, warning, shapes_list
import numpy as np
from serializable import Serializable
from semantic import SemanticResource


class Embedding(Serializable):
    name = ""
    dir_name = "embeddings"
    words = []
    dataset_embeddings = None
    embeddings = None
    words_to_numeric_idx = None
    missing = []
    embedding_dim = None
    sequence_length = None

    loaded_aggregated = False
    loaded_finalized = False

    def create(config):
        name = config.embedding.name
        if name == Train.name:
            return Train(config)
        return VectorEmbedding(config)

    def set_params(self):
        self.embedding_dim = self.config.embedding.dimension
        self.dataset_name = self.config.dataset.name
        self.aggregation = self.config.embedding.aggregation
        self.base_name = self.name
        self.sequence_length = self.config.embedding.sequence_length
        self.dataset_name = Dataset.get_limited_name(self.config)
        self.map_missing_unks = self.config.embedding.missing_words == "unk"
        if type(self.aggregation) == list:
            if self.aggregation[0] == "pad":
                self.vectors_per_doc = self.sequence_length
            elif self.aggregation[0] == "avg":
                self.vectors_per_doc = 1
            else:
                error("Undefined aggregation: {}".format(self.aggregation))
        else:
            error("Undefined aggregation: {}".format(self.aggregation))
        self.set_name()

    def save_raw_embedding_weights(self, weights):
        error("{} is for pretrained embeddings only.".format(self.name))

    def loaded_enriched(self):
        return self.loaded_finalized

    def set_name(self):
        self.name = "{}_{}_dim{}".format(self.base_name, self.dataset_name, self.embedding_dim)

    def set_raw_data_path(self):
        pass

    def __init__(self, can_fail_loading=False):
        self.set_params()
        Serializable.__init__(self, self.dir_name)
        # check for serialized mapped data
        self.set_serialization_params()
        # add paths for aggregated and enriched embeddings:
        aggr = "".join(list(map(str, self.config.embedding.aggregation + [self.sequence_length])))
        self.serialization_path_aggregated = "{}/{}.aggregated_{}.pickle".format(self.serialization_dir, self.name, aggr)
        sem = SemanticResource.get_semantic_name(self.config)
        finalized_id = sem + "_" + self.config.semantic.enrichment if sem else "nosem"
        self.serialization_path_finalized = "{}/{}.aggregated_{}.finalized_{}.pickle".format(
            self.serialization_dir, self.name, aggr, finalized_id)
        self.data_paths = [self.serialization_path_finalized, self.serialization_path_aggregated] + self.data_paths
        self.read_functions = [read_pickled]*2 + self.read_functions
        self.handler_functions = [self.handle_finalized, self.handle_aggregated] + self.handler_functions
        self.acquire2(fatal_error=not can_fail_loading)

        if self.config.semantic.disambiguation == "context_embedding":
            # need the raw embeddings even if processed embedding data is available
            if self.embeddings is None:
                self.handle_raw(self.read_functions[-1](self.get_raw_path()))


    def handle_aggregated(self, data):
        self.handle_preprocessed(data)
        self.loaded_aggregated = True
        debug("Read aggregated dataset embeddings shapes: {}, {}".format(*shapes_list(self.dataset_embeddings)))

    def handle_finalized(self, data):
        self.handle_preprocessed(data)
        self.loaded_finalized = True
        self.final_dim = data[0][0].shape[-1]
        debug("Read finalized dataset embeddings shapes: {}, {}".format(*shapes_list(self.dataset_embeddings)))

    def get_zero_pad_element(self):
        return np.zeros((1, self.embedding_dim), np.float32)

    def get_vocabulary_size(self):
        return len(self.dataset_words[0])

    def get_embeddings(self, words):
        words = [w for w in words if w in self.embeddings.index]
        word_embeddings = self.embeddings.loc[words]
        # drop the nans and return
        return word_embeddings

    def has_word(self, word):
        return word in self.embeddings.index

    def get_words(self):
        return self.words_per_document

    def get_data(self):
        return self.dataset_embeddings

    # prepare embedding data to be ready for classification
    def prepare(self):
        if self.loaded_aggregated or self.loaded_finalized:
            return
        info("Aggregating embeddings via the {} method.".format(self.aggregation))
        if self.aggregation[0] == "avg":
            # average all word vectors in the doc
            for dset_idx in range(len(self.dataset_embeddings)):
                aggregated_doc_vectors = []
                for doc_dict in self.dataset_embeddings[dset_idx]:
                    aggregated_doc_vectors.append(np.mean(doc_dict.values, axis=0))
                self.dataset_embeddings[dset_idx] = np.concatenate(aggregated_doc_vectors).reshape(
                    len(aggregated_doc_vectors), self.embedding_dim)
        elif self.aggregation[0] == "pad":
            num = self.sequence_length
            filter = self.aggregation[1]
            info("Aggregation with pad params: {} {}".format(num, filter))
            zero_pad = self.get_zero_pad_element()

            for dset_idx in range(len(self.dataset_embeddings)):
                cumulative_dataset_vectors = np.ndarray((0, self.embedding_dim), np.float32)
                num_total, num_truncated, num_padded = len(self.dataset_embeddings[dset_idx]), 0, 0
                for doc_idx in range(len(self.dataset_embeddings[dset_idx])):
                    word_vectors = self.dataset_embeddings[dset_idx][doc_idx].values
                    num_words = len(word_vectors)
                    if num_words > num:
                        # truncate
                        word_vectors = word_vectors[:num, :]
                        num_truncated += 1
                    elif num_words < num:
                        # make pad and stack vertically
                        num_to_pad = num - num_words
                        pad = np.tile(zero_pad, (num_to_pad, 1), np.float32)
                        num_padded +=1
                        word_vectors = np.append(word_vectors, pad, axis=0)
                    cumulative_dataset_vectors = np.append(cumulative_dataset_vectors, word_vectors, axis=0)
                self.dataset_embeddings[dset_idx] = cumulative_dataset_vectors
                info("Truncated {:.3f}% and padded {:.3f} % items.".format(
                    *[x / num_total * 100 for x in [num_truncated, num_padded]]))

        # serialize aggregated
        write_pickled(self.serialization_path_aggregated, self.get_all_preprocessed())

    # finalize embeddings to use for training, aggregating all data to a single ndarray
    # if semantic enrichment is selected, do the infusion
    def finalize(self, semantic):
        if self.loaded_finalized:
            info("Skipping embeddings finalizing, since finalized data was already loaded.")
            return
        finalized_name = self.name
        if self.config.semantic.enrichment is not None:
            if self.config.embedding.name == "train":
                error("Semantic enrichment undefined for embedding training, for now.")
            info("Enriching {} embeddings with semantic information, former having {} vecs/doc.".format(self.config.embedding.name, self.vectors_per_doc))
            semantic_data = semantic.get_vectors()
            finalized_name += ".{}.enriched".format(SemanticResource.get_semantic_name(self.config))

            if self.config.semantic.enrichment == "concat":
                semantic_dim = len(semantic_data[0][0])
                self.final_dim = self.embedding_dim + semantic_dim
                for dset_idx in range(len(semantic_data)):
                    info("Concatenating dataset part {}/{} to composite dimension: {}".format(dset_idx+1, len(semantic_data), self.final_dim))
                    if self.vectors_per_doc > 1:
                        # tile the vector the needed times to the right, reshape to the correct dim
                        semantic_data[dset_idx] = np.reshape(np.tile(semantic_data[dset_idx], (1, self.vectors_per_doc)),
                                                             (-1, semantic_dim))
                    self.dataset_embeddings[dset_idx] = np.concatenate(
                        [self.dataset_embeddings[dset_idx], semantic_data[dset_idx]], axis=1)


            elif self.config.semantic.enrichment == "replace":
                self.final_dim = len(semantic_data[0][0])
                for dset_idx in range(len(semantic_data)):
                    info("Replacing dataset part {}/{} with semantic info of dimension: {}".format(dset_idx+1, len(semantic_data), self.final_dim))
                    if self.vectors_per_doc > 1:
                        # tile the vector the needed times to the right, reshape to the correct dim
                        semantic_data[dset_idx] = np.reshape(np.tile(semantic_data[dset_idx], (1, self.vectors_per_doc)),
                                                             (-1, self.final_dim))
                    self.dataset_embeddings[dset_idx] = semantic_data[dset_idx]
            else:
                error("Undefined semantic enrichment: {}".format(self.config.semantic.enrichment))
        else:
            info("Finalizing embeddings without semantic information.")
            finalized_name += ".finalized"
            dim = self.embedding_dim if not self.config.embedding.name == "train" else 1
            self.final_dim = dim
            # concatenating embeddings for each dataset portion into a single dataframe
            for dset_idx in range(len(self.dataset_embeddings)):
                new_dset_embeddings = np.ndarray((0, dim), np.float32)
                for doc_idx in range(len(self.dataset_embeddings[dset_idx])):
                    embeddings = self.dataset_embeddings[dset_idx][doc_idx]
                    new_dset_embeddings = np.vstack([new_dset_embeddings, embeddings])
                self.dataset_embeddings[dset_idx] = new_dset_embeddings

        # serialize finalized embeddings
        write_pickled(self.serialization_path_finalized, self.get_all_preprocessed())


    def get_final_dim(self):
        return self.final_dim

    def get_dim(self):
        return self.embedding_dim

    def preprocess(self):
        error("Need to override embedding preprocessing for {}".format(self.name))

# generic class to load pickled embedding dataframes
class VectorEmbedding(Embedding):
    name = "vector"
    unknown_word_token = "unk"

    # expected raw data path
    def get_raw_path(self):
        return "{}/{}_dim{}.pickle".format(self.raw_data_dir, self.base_name, self.embedding_dim)

    # mark word-index relations for stat computation, add unk if needed
    def handle_raw_serialized(self, raw_serialized):
        # process as dataframe
        self.words_to_numeric_idx = {}
        self.embeddings = raw_serialized
        for w in self.embeddings.index.tolist():
            self.words_to_numeric_idx[w] = len(self.words_to_numeric_idx)

    # mark preprocessing
    def handle_preprocessed(self, preprocessed):
        self.dataset_embeddings, self.words_per_document, self.missing,  \
        self.undefined_word_index, self.present_word_indexes = preprocessed
        self.loaded_preprocessed = True
        debug("Read preprocessed dataset embeddings shapes: {}, {}".format(*list(map(len, self.dataset_embeddings))))

    def handle_raw(self, raw_data):
        self.handle_raw_serialized(raw_data)

    def fetch_raw(self, path):
        # assume embeddings are dataframes
        return read_pickled(path)

    def preprocess(self):
        pass

    # transform input texts to embeddings
    def map_text(self, dset):
        if self.loaded_preprocessed or self.loaded_aggregated or self.loaded_finalized:
            return
        info("Mapping {} to {} embeddings.".format(dset.name, self.name))
        text_bundles = dset.train, dset.test
        self.dataset_embeddings = []
        self.words_per_document = []
        self.present_word_indexes = []
        self.vocabulary = dset.vocabulary


        if self.unknown_word_token not in self.embeddings and self.map_missing_unks:
            warning("[{}] unknown token missing from embeddings, adding it as zero vector.".format(self.unknown_word_token))
            self.embeddings.loc[self.unknown_word_token] = np.zeros(self.embedding_dim)

        embedded_words, unknown_token = self.embeddings.index.values, self.unknown_word_token
        # loop over input text bundles (e.g. train & test)
        for i in range(len(text_bundles)):
            self.dataset_embeddings.append([])
            self.words_per_document.append([])
            self.present_word_indexes.append([])
            with tictoc("Embedding mapping for text bundle {}/{}".format(i+1, len(text_bundles))):
                info("Mapping text bundle {}/{}: {} texts".format(i+1, len(text_bundles), len(text_bundles[i])))
                hist = {w: 0 for w in embedded_words}
                hist_missing = {}
                for j, word_list in enumerate(text_bundles[i]):
                    debug("Text {}/{} with {} words".format(j+1, len(text_bundles[i]), len(word_list)))
                    # check present & missing words
                    missing_words, missing_index, present_words, present_index = [], [], [], []
                    for w, word in enumerate(word_list):
                        if word not in embedded_words:
                            missing_words.append(word)
                            missing_index.append(w)
                            if word not in hist_missing:
                                hist_missing[word] = 0
                            hist_missing[word] += 1
                        else:
                            present_words.append(word)
                            present_index.append(w)
                            hist[word] +=1

                    # handle missing
                    if not self.map_missing_unks:
                        # ignore & discard missing words
                        word_list = present_words
                    else:
                        # map missing to UNKs
                        for m in missing_index:
                            word_list[m] = self.unknown_word_token

                    # get embeddings
                    text_embeddings = self.embeddings.loc[word_list]
                    self.dataset_embeddings[-1].append(text_embeddings)

                    # update present words and their index, per doc
                    self.words_per_document[-1].append(present_words)
                    self.present_word_indexes[-1].append(present_index)

            self.print_word_stats(hist, hist_missing)
            self.missing.append(hist_missing)

        # write
        info("Writing embedding mapping to {}".format(self.serialization_path_preprocessed))
        write_pickled(self.serialization_path_preprocessed, self.get_all_preprocessed())
        # log missing words
        missing_filename = self.serialization_path_preprocessed + ".missingwords"
        write_pickled(missing_filename, self.missing)

    def print_word_stats(self, hist, hist_missing):
        num_words_hit, num_hit = sum([1 for v in hist if hist[v] > 0]), sum(hist.values())
        num_words_miss, num_miss = len(hist_missing.keys()), sum(hist_missing.values())
        num_total = sum(list(hist.values()) + list(hist_missing.values()))

        debug("Found {} instances or {:.3f} % of total {}, for {} words.".format(num_hit, num_hit/num_total*100, num_total, num_words_hit))
        debug("Missed {} instances or {:.3f} % of total {}, for {} words.".format(num_miss, num_miss/num_total*100, num_total, num_words_miss))


    def __init__(self, config):
        self.config = config
        self.name = self.base_name = self.config.embedding.name
        Embedding.__init__(self)

    def get_all_preprocessed(self):
        return [self.dataset_embeddings, self.words_per_document, self.missing, None, self.present_word_indexes]

    def get_present_word_indexes(self):
        return self.present_word_indexes

class Train(Embedding):
    name = "train"
    undefined_word_name = "unk"

    def __init__(self, config):
        self.config = config
        Embedding.__init__(self, can_fail_loading=True)


    # embedding training data (e.g. word indexes) does not depend on embedding dimension
    # so naming is overriden to omit embedding dimension
    def set_name(self):
        self.name = "{}_{}".format(self.base_name, self.dataset_name)

    # transform input texts to embeddings
    def map_text(self, dset):
        # assign all embeddings
        self.embeddings = pd.DataFrame(dset.vocabulary_index, dset.vocabulary)
        if self.loaded_preprocessed or self.loaded_aggregated or self.loaded_finalized:
            return
        info("Mapping {} to {} embeddings.".format(dset.name, self.name))
        text_bundles = dset.train, dset.test
        self.dataset_embeddings = []
        self.undefined_word_index = dset.undefined_word_index
        non_train_words = []
        # loop over input text bundles (e.g. train & test)
        for i in range(len(text_bundles)):
            self.dataset_embeddings.append([])
            with tictoc("Embedding mapping for text bundle {}/{}".format(i+1, len(text_bundles))):
                info("Mapping text bundle {}/{}: {} texts".format(i+1, len(text_bundles), len(text_bundles[i])))
                for j in range(len(text_bundles[i])):
                    word_list = text_bundles[i][j]
                    index_list = [ [dset.word_to_index[w]] if w in dset.vocabulary else [dset.undefined_word_index] for w in word_list]
                    embedding = pd.DataFrame(index_list, index = word_list)
                    debug("Text {}/{}".format(j+1, len(text_bundles[i])))
                    self.dataset_embeddings[-1].append(embedding)
                    # get test words, perhaps
                    if i > 0:
                        for w in word_list:
                            if w not in non_train_words:
                                non_train_words.append(w)
        self.dataset_words = [dset.vocabulary, non_train_words]
        # write mapped data
        write_pickled(self.serialization_path_preprocessed, self.get_all_preprocessed())

    def get_all_preprocessed(self):
        return [self.dataset_embeddings, self.dataset_words, None, self.undefined_word_index, None]

    def get_zero_pad_element(self):
        return self.undefined_word_index

    def get_raw_path(self):
        return None

    def fetch_raw(self, dummy_input):
        return dummy_input

    def handle_preprocessed(self, preprocessed):
        self.loaded_preprocessed = True
        self.dataset_embeddings, self.dataset_words, self.missing, self.undefined_word_index, _ = preprocessed

    def save_raw_embedding_weights(self, weights, write_dir):
        # rename to generic vectorembedding
        emb_name = ("raw_" + self.name + "_dim{}.pickle".format(self.config.embedding.dimension)).replace(Train.name, VectorEmbedding.name)
        writepath = join(write_dir, emb_name)
        # associate with respective words
        index = self.dataset_words[0] + [self.undefined_word_name]
        data = pd.DataFrame(weights, index = index)
        write_pickled(writepath, data)

    def get_words(self):
        return self.dataset_words


class Universal_sentence_encoder:
    pass

class ELMo:
    # https://allennlp.org/elmo
    pass

class Word2vec(Embedding):
    name = "word2vec"
    def __init__(self, params):
        Embedding.__init__(self)

class FastText:
    # https://fasttext.cc/docs/en/english-vectors.html
    pass
class Doc2vec:
    pass
