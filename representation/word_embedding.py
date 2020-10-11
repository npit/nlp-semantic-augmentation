import math
from collections import Counter

import numpy as np
import pandas as pd
import tqdm

import defs

from representation.embedding import Embedding
import collections
from utils import (debug, error, info, realign_embedding_index, tictoc,
                   warning, write_pickled)


# generic class to load pickled embedding vectors
class WordEmbedding(Embedding):
    name = "word_embedding"
    unknown_word_token = "unk"
    present_words = None

    data_names = Embedding.data_names + ["unknown_element_index"]

    # expected raw data path
    def get_raw_path(self):
        return "{}/{}_dim{}.pkl".format(self.raw_data_dir, self.base_name, self.dimension)

    def map_text_partial_load(self):
        # iterate over files. match existing words
        # map embedding vocab indexes to files and word positions in that file
        # iterate sequentially the csv with batch loading
        batch_size = 10000
        info("Mapping with a read batch size of {}".format(batch_size))
        error("Partial loading implementation has to include UNKs", not self.map_missing_unks)

        # make a mapping from the embedding vocabulary index -> file_index, word_index
        # thus the embeddings file can be traversed just once
        vocab_to_dataset = {}

        for dset_idx, docs in enumerate(self.text):
            num_docs = len(docs)
            current_num_words = 0
            word_stats = WordEmbeddingStats(self.vocabulary, self.embedding_vocabulary_index.keys())
            # make empty features
            with tqdm.tqdm("Bulding embedding / dataset vocabulary mapping for text bundle {}/{}".format(dset_idx + 1, len(self.text)),
                           total=num_docs, ascii=True) as pbar:
                for file_idx, word_info_list in enumerate(self.text[dset_idx]):
                    for word_idx, word_info in enumerate(word_info_list):
                        word = word_info[0]
                        word_in_embedding_vocab = word in self.embedding_vocabulary_index
                        word_stats.update_word_stats(word, word_in_embedding_vocab)
                        if word_in_embedding_vocab:
                            # word exists in the embeddings vocabulary
                            vocab_index = self.embedding_vocabulary_index[word]
                            if vocab_index not in vocab_to_dataset:
                                vocab_to_dataset[vocab_index] = []
                            vocab_to_dataset[vocab_index].append((dset_idx, current_num_words))
                            self.present_words.add(word)
                        current_num_words += 1
                    pbar.update()
                    self.elements_per_instance[dset_idx].append(len(word_info_list))
            word_stats.print_word_stats()
            self.vector_indices[dset_idx] = np.repeat(self.get_zero_pad_element(), current_num_words, axis=0)
            self.elements_per_instance[dset_idx] = np.asarray(self.elements_per_instance[dset_idx], np.int32)

        # get the embedding indexes words are mapped to
        present_embedding_word_indexes = sorted(vocab_to_dataset.keys(), reverse=True)
        # self.mapped_embedding_words = set([self.embedding_vocabulary_index[i] for i in set(present_embedding_word_indexes)])

        # populate features by batch-loading the embedding csv
        num_chunks = math.ceil(len(self.embedding_vocabulary_index) / batch_size)
        with tqdm.tqdm(total=num_chunks, ascii=True) as pbar:
            for chunk_index, chunk in enumerate(pd.read_csv(self.embeddings_path, index_col=0, header=None,
                                                delim_whitespace=True, chunksize=batch_size)):
                while True:
                    if not present_embedding_word_indexes:
                        break
                    # get next vocabulary index of interest
                    vocab_index = present_embedding_word_indexes.pop()
                    if vocab_index >= (chunk_index + 1) * batch_size:
                        # vocab index is in the next chunk. reinsert and continue
                        present_embedding_word_indexes.append(vocab_index)
                        break
                    # get local (in the batch) index position
                    batch_vocab_index = vocab_index - batch_size * chunk_index
                    vector = chunk.iloc[batch_vocab_index].values
                    for dset_idx, global_word_idx in vocab_to_dataset[vocab_index]:
                        self.vector_indices[dset_idx][global_word_idx] = vector
                pbar.set_description(desc="Chunk {}/{}".format(chunk_index + 1, num_chunks))
                pbar.update()

        # write
        info("Writing embedding mapping to {}".format(self.serialization_path_preprocessed))
        write_pickled(self.serialization_path_preprocessed, self.get_all_preprocessed())

    # mark preprocessing
    def handle_preprocessed(self, preprocessed):
        self.undefined_word_index = preprocessed["undefined_element_index"]
        Embedding.handle_preprocessed(self, preprocessed)

    def read_raw_embedding_mapping(self, path):
        super().read_raw_embedding_mapping(path)
        if self.unknown_word_token not in self.embeddings_source:
            self.embeddings_source.loc[self.unknown_word_token] = np.zeros(self.dimension)

    def produce_outputs(self):
        """Produce word embeddings"""
        # will not limit the embedding source to the used embeddings,
        # since the model may be used for testing

        # make index to position map to speed up location finder
        self.key2pos_map = {}
        counter = collections.Counter()
        for ind in tqdm.tqdm(self.embeddings_source.index, total=len(self.embeddings_source), desc="Buildilng key index map"):
            self.key2pos_map[ind] = len(self.key2pos_map)

        self.embeddings, self.elements_per_instance = [], []
        for doc_dict in tqdm.tqdm(self.text.data.instances, total=len(self.text.data.instances), desc="Mapping word embeddings"):
            words = doc_dict['words']
            self.map_words(words, self.embeddings)

    def map_words(self, words, embeddings):
        """Map input words to indexes of the read embeddings source"""

        # idxs = [self.embeddings_source.index.get_loc(w) if w in self.embeddings_source.index \
        #         else self.embeddings_source.index.get_loc(self.unknown_word_token) \
        #         for w in words]

        idxs = [self.key2pos_map[w] if w in self.key2pos_map else self.key2pos_map[self.unknown_word_token] for w in words]
        if not idxs:
            idxs = [self.key2pos_map(self.unknown_word_token)]
        # self.embeddings = np.ndarray((0, self.dimension), dtype=np.float32)
        self.elements_per_instance.append(len(idxs))

        if self.aggregation == defs.aggregation.avg:
            # we have to produce a new embedding
            vec = np.mean(self.embeddings_source.iloc[idxs].values, axis=0)
            self.embeddings.append(vec)

        elif self.aggregation == defs.aggregation.pad:
            num_vectors = len(idxs)
            if self.sequence_length < num_vectors:
                # truncate
                idxs = idxs[:self.sequence_length]
            elif self.sequence_length > num_vectors:
                # make pad and stack vertically
                pad_size = self.sequence_length - num_vectors
                idxs += [self.unknown_element_index] * pad_size
            vecs = self.embeddings_source[idxs]
            self.embeddings.append(vecs)
        else:
            error(f"Undefined aggregation {self.aggregation}")

    # transform input texts to embeddings
    def map_text(self):
        if self.loaded_aggregated:
            return
        if self.loaded_preprocessed or self.loaded_aggregated:
            self.aggregate_instance_vectors()
            return
        info("Mapping to {} word embeddings.".format(self.name))

        self.vector_indices = [[], []]
        self.indices = [[], []]
        self.elements_per_instance = [[], []]
        self.present_words = set()

        # if there's a vocabulary file read or precomputed, utilize partial loading of the underlying csv
        # saves a lot of memory
        if self.embedding_vocabulary_index:
            self.map_text_partial_load()
        else:
            self.map_text_nonpartial_load()
        self.aggregate_instance_vectors()

    def map_text_nonpartial_load(self):
        # initialize unknown token embedding, if it's not defined
        if self.unknown_word_token not in self.embeddings and self.map_missing_unks:
            warning("[{}] unknown token missing from embeddings, adding it as zero vector.".format(self.unknown_word_token))
            self.embeddings.loc[self.unknown_word_token] = np.zeros(self.dimension)

        unknown_token_index = self.embeddings.index.get_loc(self.unknown_word_token)
        self.unknown_element_index = unknown_token_index
        # loop over input text bundles
        for dset_idx, docs in enumerate(self.text):
            num_docs = len(docs)
            desc = "Embedding mapping for text bundle {}/{}, with {} texts".format(dset_idx + 1, len(self.text), num_docs)
            with tqdm.tqdm(total=len(self.text[dset_idx]), ascii=True, desc=desc) as pbar:
                word_stats = WordEmbeddingStats(self.vocabulary, self.embeddings.index)
                for j, doc_wp_list in enumerate(self.text[dset_idx]):
                    # drop POS
                    word_list = [wp[0] for wp in doc_wp_list]
                    # debug("Text {}/{} with {} words".format(j + 1, num_documents, len(word_list)))
                    # check present & missing words
                    doc_indices = []
                    present_map = {}
                    present_index_map = {}
                    for w, word in enumerate(word_list):
                        word_in_embedding_vocab = word in self.embeddings.index
                        word_stats.update_word_stats(word, word_in_embedding_vocab)
                        present_map[word] = word_in_embedding_vocab
                        present_map[word] = word_in_embedding_vocab
                        if not word_in_embedding_vocab:
                            if not self.map_missing_unks:
                                continue
                            else:
                                word_index = unknown_token_index

                        else:
                            word_index = self.embeddings.index.get_loc(word)
                        doc_indices.append(word_index)
                    # handle missing
                    word_list = [w for w in word_list if present_map[w]] if not self.map_missing_unks else \
                        [w if present_map[w] else self.unknown_word_token for w in word_list]

                    self.present_words.update([w for w in present_map if present_map[w]])
                    error("No words present in document.", len(word_list) == 0)

                    # just save indices
                    doc_indices = np.asarray(doc_indices, np.int32)
                    self.vector_indices[dset_idx].append(doc_indices)
                    self.elements_per_instance[dset_idx].append(len(doc_indices))
                    pbar.update()

            word_stats.print_word_stats()
            self.elements_per_instance[dset_idx] = np.asarray(self.elements_per_instance[dset_idx], np.int32)

        self.vector_indices, new_embedding_index = realign_embedding_index(self.vector_indices, np.asarray(list(range(len(self.embeddings.index)))))
        self.embeddings = self.embeddings.iloc[new_embedding_index].values
        # write
        info("Writing embedding mapping to {}".format(self.serialization_path_preprocessed))
        write_pickled(self.serialization_path_preprocessed, self.get_all_preprocessed())


    # getter for semantic processing, filtering only to present words
    def process_data_for_semantic_processing(self, train, test):
        """Deprecated"""
        if self.map_missing_unks:
            return Embedding.process_data_for_semantic_processing(self, train, test)
        # if we discard words that are not in the vocabulary, return those that are in the vocabulary
        # s.t. semantic processing is only applied on them
        ret_train, ret_test = [[w for w in doc if w[0] in self.present_words] for doc in train], \
                              [[w for w in doc if w[0] in self.present_words] for doc in test]
        return ret_train, ret_test


    def __init__(self, config):
        """Constructor for the word embeddings class"""
        self.config = config
        self.name = self.base_name = self.config.name
        Embedding.__init__(self)


class WordEmbeddingStats:
    """Class to compute and show word-embedding matching / coverage statistics"""

    def __init__(self, dataset_vocabulary, embedding_vocab):
        self.hist = {w: 0 for w in embedding_vocab}
        self.hist_missing = Counter()
        self.dataset_vocabulary = dataset_vocabulary

    def update_word_stats(self, word, word_in_embedding_vocabulary):
        if not word_in_embedding_vocabulary:
            self.hist_missing[word] += 1
        else:
            self.hist[word] += 1

    def print_word_stats(self):
        try:
            terms_hit, hit_sum = len([v for v in self.hist if self.hist[v] > 0]), sum(self.hist.values())
            terms_missed, miss_sum = len([v for v in self.hist_missing if self.hist_missing[v] > 0]), \
                                     sum(self.hist_missing.values())
            total_term_sum = sum(list(self.hist.values()) + list(self.hist_missing.values()))
            debug("{:.3f} % terms in the vocabulary appear at least once, which corresponds to a total of {:.3f} % terms in the text".
                  format(terms_hit / len(self.hist) * 100, hit_sum / total_term_sum * 100))
            debug("{:.3f} % terms in the vocabulary never appear, i.e. a total of {:.3f} % terms in the text".format(
                terms_missed / len(self.dataset_vocabulary) * 100, miss_sum / total_term_sum * 100))
        except ZeroDivisionError:
            warning("No samples!")
