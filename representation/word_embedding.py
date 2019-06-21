import pandas as pd

from representation.embedding import Embedding
from utils import error, info, warning, tictoc, write_pickled, debug
import numpy as np


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
        error("Partial map text is todo,")

    # transform input texts to embeddings
    def map_text(self, dset):
        if self.loaded_preprocessed or self.loaded_aggregated or self.loaded_finalized:
            return
        info("Mapping dataset: {} to {} embeddings.".format(dset.name, self.name))

        # if there's a vocabulary file read or precomputed, utilize partial loading of the underlying csv
        # saves a lot of memory
        if self.embedding_vocabulary_index:
            self.map_text_partial_load(dset)
            return

        text_bundles = dset.train, dset.test
        self.dataset_vectors = [[], []]
        self.present_term_indexes = [[], []]
        self.vocabulary = dset.vocabulary
        self.elements_per_instance = [[], []]

        # initialize unknown token embedding, if it's not defined
        if self.unknown_word_token not in self.embeddings and self.map_missing_unks:
            warning("[{}] unknown token missing from embeddings, adding it as zero vector.".format(self.unknown_word_token))
            self.embeddings.loc[self.unknown_word_token] = np.zeros(self.dimension)


        # loop over input text bundles (e.g. train & test)
        for dset_idx in range(len(text_bundles)):
            with tictoc("Embedding mapping for text bundle {}/{}".format(dset_idx + 1, len(text_bundles))):
                info("Mapping text bundle {}/{}: {} texts".format(dset_idx + 1, len(text_bundles), len(text_bundles[dset_idx])))
                hist = {w: 0 for w in self.embeddings.index}
                hist_missing = {}
                num_documents = len(text_bundles[dset_idx])
                for j, doc_wp_list in enumerate(text_bundles[dset_idx]):
                    # drop POS
                    word_list = [wp[0] for wp in doc_wp_list]
                    # debug("Text {}/{} with {} words".format(j + 1, num_documents, len(word_list)))
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
                        # print(word_list)

                    if not present_terms and not self.map_missing_unks:
                        # no words present in the mapping, force
                        error("No words persent in document.")

                    # get embeddings
                    text_embeddings = self.embeddings.loc[word_list]
                    self.dataset_vectors[dset_idx].append(text_embeddings)

                    # update present words and their index, per doc
                    num_embeddings = len(text_embeddings)
                    if num_embeddings == 0:
                        error("No embeddings generated for text")
                    self.elements_per_instance[dset_idx].append(num_embeddings)
                    self.present_term_indexes[dset_idx].append(present_index)

            if len(self.dataset_vectors[dset_idx]) > 0:
                self.dataset_vectors[dset_idx] = pd.concat(self.dataset_vectors[dset_idx]).values
            self.print_word_stats(hist, hist_missing)

        # write
        info("Writing embedding mapping to {}".format(self.serialization_path_preprocessed))
        write_pickled(self.serialization_path_preprocessed, self.get_all_preprocessed())

    def print_word_stats(self, hist, hist_missing):
        try:
            terms_hit, hit_sum = len([v for v in hist if hist[v] > 0]), sum(hist.values())
            terms_missed, miss_sum = len([v for v in hist_missing if hist_missing[v] > 0]), \
                                     sum(hist_missing.values())
            total_term_sum = sum(list(hist.values()) + list(hist_missing.values()))
            debug("{:.3f} % terms in the vocabulary appear at least once, which corresponds to a total of {:.3f} % terms in the text".
                  format(terms_hit / len(hist) * 100, hit_sum / total_term_sum * 100))
            debug("{:.3f} % terms in the vocabulary never appear, i.e. a total of {:.3f} % terms in the text".format(
                terms_missed / len(self.vocabulary) * 100, miss_sum / total_term_sum * 100))
        except ZeroDivisionError:
            warning("No samples!")

    def __init__(self, config):
        self.config = config
        self.name = self.base_name = self.config.representation.name
        Embedding.__init__(self)
