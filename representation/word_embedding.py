import pandas as pd
import math
from representation.embedding import Embedding
from utils import error, info, warning, tictoc, write_pickled, debug
import numpy as np
import tqdm


# generic class to load pickled embedding vectors
class WordEmbedding(Embedding):
    name = "word_embedding"
    unknown_word_token = "unk"


    # expected raw data path
    def get_raw_path(self):
        return "{}/{}_dim{}.pickle".format(self.raw_data_dir, self.base_name, self.dimension)


    def map_text_partial_load(self, dset):
        # iterate over files. match existing words
        # map embedding vocab indexes to files and word positions in that file
        # iterate sequentially the csv with batch loading
        text_bundles = dset.train, dset.test
        batch_size = 10000
        info("Mapping with a read batch size of {}".format(batch_size))
        error("Partial loading implementation has to include UNKs", not self.map_missing_unks)

        # make a mapping embedding vocab index -> file_index, word_index
        vocab_to_dataset = {}

        for dset_idx, docs in enumerate(text_bundles):
            num_docs = len(docs)
            current_num_words = 0
            word_stats = WordEmbeddingStats(self.vocabulary, self.embedding_vocabulary_index.keys())
            # make empty features
            with tqdm.tqdm("Bulding embedding / dataset vocabulary mapping for text bundle {}/{}".format(dset_idx + 1, len(text_bundles)),
                           total=num_docs, ascii=True) as pbar:
                for file_idx, word_info_list in enumerate(text_bundles[dset_idx]):
                    for word_idx, word_info in enumerate(word_info_list):
                        word = word_info[0]
                        word_in_vocab = word in self.embedding_vocabulary_index
                        word_stats.update_word_stats(word, word_in_vocab)
                        if word_in_vocab:
                            # word exists in the embeddings vocabulary
                            vocab_index = self.embedding_vocabulary_index[word]
                            if vocab_index not in vocab_to_dataset:
                                vocab_to_dataset[vocab_index] = []
                            vocab_to_dataset[vocab_index].append((dset_idx, current_num_words))
                        current_num_words += 1
                    pbar.update()
                    self.elements_per_instance[dset_idx].append(len(word_info_list))
            word_stats.print_word_stats()
            self.dataset_vectors[dset_idx] = np.repeat(self.get_zero_pad_element(), current_num_words, axis=0)

        useful_indexes = sorted(vocab_to_dataset.keys(), reverse=True)

        # populate features by batch-loading the embedding csv
        num_chunks = math.ceil(len(self.embedding_vocabulary_index) / batch_size)
        with tqdm.tqdm(total=num_chunks, ascii=True) as pbar:
            for chunk_index, chunk in enumerate(pd.read_csv(self.embeddings_path, index_col=0, header=None,
                                                delim_whitespace=True, chunksize=batch_size)):
                while True:
                    if not useful_indexes:
                        break
                    # get next vocabulary index of interest
                    vocab_index = useful_indexes.pop()
                    if vocab_index >= (chunk_index + 1) * batch_size:
                        # vocab index is in the next chunk. reinsert and continue
                        useful_indexes.append(vocab_index)
                        break
                    # get local (in the batch) index position
                    batch_vocab_index = vocab_index - batch_size * chunk_index
                    vector = chunk.iloc[batch_vocab_index].values
                    for dset_idx, global_word_idx in vocab_to_dataset[vocab_index]:
                        self.dataset_vectors[dset_idx][global_word_idx] = vector
                pbar.set_description(desc="Chunk {}/{}".format(chunk_index+1, num_chunks))
                pbar.update()

        # write
        info("Writing embedding mapping to {}".format(self.serialization_path_preprocessed))
        write_pickled(self.serialization_path_preprocessed, self.get_all_preprocessed())

    # transform input texts to embeddings
    def map_text(self, dset):
        if self.loaded_preprocessed or self.loaded_aggregated or self.loaded_finalized:
            return
        info("Mapping dataset: {} to {} embeddings.".format(dset.name, self.name))

        self.dataset_vectors = [[], []]
        self.elements_per_instance = [[], []]
        self.vocabulary = dset.vocabulary

        # if there's a vocabulary file read or precomputed, utilize partial loading of the underlying csv
        # saves a lot of memory
        if self.embedding_vocabulary_index:
            self.map_text_partial_load(dset)
            return

        text_bundles = dset.train, dset.test

        # initialize unknown token embedding, if it's not defined
        if self.unknown_word_token not in self.embeddings and self.map_missing_unks:
            warning("[{}] unknown token missing from embeddings, adding it as zero vector.".format(self.unknown_word_token))
            self.embeddings.loc[self.unknown_word_token] = np.zeros(self.dimension)

        # loop over input text bundles (e.g. train & test)
        for dset_idx, docs in enumerate(text_bundles):
            num_docs = len(docs)
            with tictoc("Embedding mapping for text bundle {}/{}, with {} texts".format(dset_idx + 1, len(text_bundles), num_docs)):
                word_stats = WordEmbeddingStats(self.vocabulary, self.embeddings.index)
                for j, doc_wp_list in enumerate(text_bundles[dset_idx]):
                    # drop POS
                    word_list = [wp[0] for wp in doc_wp_list]
                    # debug("Text {}/{} with {} words".format(j + 1, num_documents, len(word_list)))
                    # check present & missing words
                    present_map = {}
                    for w, word in enumerate(word_list):
                        word_in_embedding_vocab = word in self.embeddings.index
                        word_stats.update_word_stats(word, word_in_embedding_vocab)
                        present_map[word] = word_in_embedding_vocab

                    # handle missing
                    word_list = [w for w in word_list if present_map[w]] if not self.map_missing_unks else \
                        [w if present_map[w] else self.unknown_word_token for w in word_list]

                    error("No words present in document.", len(word_list) == 0)

                    # get embeddings
                    text_embeddings = self.embeddings.loc[word_list]
                    self.dataset_vectors[dset_idx].append(text_embeddings)

                    # update present words and their index, per doc
                    num_embeddings = len(text_embeddings)
                    error("No embeddings generated for text", num_embeddings == 0)
                    self.elements_per_instance[dset_idx].append(num_embeddings)

            if len(self.dataset_vectors[dset_idx]) > 0:
                self.dataset_vectors[dset_idx] = pd.concat(self.dataset_vectors[dset_idx]).values

            word_stats.print_word_stats()

        # write
        info("Writing embedding mapping to {}".format(self.serialization_path_preprocessed))
        write_pickled(self.serialization_path_preprocessed, self.get_all_preprocessed())



    def __init__(self, config):
        self.config = config
        self.name = self.base_name = self.config.representation.name
        Embedding.__init__(self)


class WordEmbeddingStats:
    """Class to compute and show word-embedding matching / coverage statistics"""

    def __init__(self, vocabulary, embedding_vocab):
        self.hist = {w: 0 for w in embedding_vocab}
        self.hist_missing = {}
        self.vocabulary = vocabulary

    def update_word_stats(self, word, word_in_embedding_vocabulary):
        if not word_in_embedding_vocabulary:
            if word not in self.hist_missing:
                self.hist_missing[word] = 0
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
                terms_missed / len(self.vocabulary) * 100, miss_sum / total_term_sum * 100))
        except ZeroDivisionError:
            warning("No samples!")
