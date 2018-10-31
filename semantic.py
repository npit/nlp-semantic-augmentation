# import gensim
from os.path import join, exists
from os import makedirs
import pickle
from utils import read_pickled, write_pickled
from utils import tic, toc, error, info, debug, warning, write_pickled
import numpy as np
from serializable import Serializable
from nltk.corpus import wordnet as wn


class SemanticResource(Serializable):
    serialization_subdir = "semantic"

    def __init__(self, config):
        Serializable.__init__(self, join(config.folders.serialization, self.serialization_subdir))

    def create(config):
        name = config.semantic.name
        if name == Wordnet.name:
            return Wordnet(config)
        else:
            error("Undefined semantic resource: {}".format(name))
    pass


    name = "wordnet"
class Wordnet(SemanticResource):
    serialization_dir = "serializations/semantic_data"
    word_synset_lookup_cache = {}
    word_synset_embedding_cache = {}

    synset_freqs = []
    synset_tfidf_freqs = []
    dataset_freqs = []
    dataset_minmax_freqs = []
    assignments = {}
    synset_context_word_threshold = None

    reference_synsets = None

    def __init__(self, config):
        SemanticResource.__init__(self, config)
        self.config = config

        self.base_name = self.name

        self.serialization_dir = join(config.folders.serialization, "semantic_data")
        if not exists(self.serialization_dir):
            makedirs(self.serialization_dir, exist_ok=True)
        self.semantic_freq_threshold = config.semantic.frequency_threshold
        self.semantic_weights = config.semantic.weights
        self.semantic_unit = config.semantic.unit
        self.disambiguation = config.semantic.disambiguation
        self.spreading_activation = config.semantic.spreading_activation

        self.dataset_name = config.dataset.name
        if config.dataset.limit:
            self.dataset_name += "_{}".format(config.dataset.limit)

        freq_filtering = "ALL" if not self.semantic_freq_threshold else "fthres{}".format(self.semantic_freq_threshold)
        sem_weights = "w{}".format(self.semantic_weights)
        disambig = "disam{}".format(self.disambiguation)
        self.name = "{}_{}_{}_{}_{}".format(self.dataset_name, self.name, sem_weights, freq_filtering, disambig)

        if self.disambiguation == "context-embedding":
            self.semantic_embedding_dim = self.config.embedding.dimension
            # incompatible with embedding training
            if config.embedding.name == "train":
                error("Embedding train mode incompatible with semantic embeddings.")
            # preload the concept list
            context_file = config.semantic.context_file
            with open(context_file, "rb") as f:
                data = pickle.load(f)
                self.semantic_context = {}
                if self.synset_context_word_threshold is not None:
                    info("Limiting reference context synsets to a frequency threshold of {}".format(self.synset_context_word_threshold))
                    for s, wl in data.items():
                        if len(wl) < self.synset_context_word_threshold:
                            continue
                        self.semantic_context[s] = wl
                else:
                    self.semantic_context = data;
                self.reference_synsets = list(self.semantic_context.keys())
                if not self.reference_synsets:
                    error("Frequency threshold of synset context: {} resulted in zero reference synsets".format(self.synset_context_word_threshold))
                info("Applying {} reference synsets from pre-extracted synset words".format(len(self.reference_synsets)))
            # calculate the synset embeddings path
            thr = ""
            if self.synset_context_word_threshold:
                thr += "_thresh{}".format(self.synset_context_word_threshold)
            self.sem_embeddings_path = join(self.serialization_dir, "semantic_embeddings_{}_{}{}{}".format(self.name, self.embedding.name, self.semantic_embedding_dim, thr))
            self.semantic_embedding_aggr = config.semantic.context_aggregation

        # setup serialization paramas
        self.set_serialization_params()

        # load if existing
        self.acquire2(fatal_error=False)



    def fetch_raw(self, dummy_input):
        # set name and paths to load sem. emb.
        # or set a sem. emb. serializable class
        if self.disambiguation == "context-embedding":
            self.compute_semantic_embeddings()

    def handle_preprocessed(self, preprocessed):
        self.loaded_preprocessed = True
        self.assignments, self.synset_freqs, self.dataset_freqs, self.synset_tfidf_freqs, self.reference_synsets, self.present_word_indexes = preprocessed

    def handle_raw_serialized(self, raw_serialized):
        pass
    def handle_raw(self, raw_data):
        pass

    # prune semantic information units wrt a frequency threshold
    def apply_freq_filtering(self, freq_dict_list, dataset_freqs, force_reference=False):
        info("Applying synset frequency filtering with a threshold of {}".format(self.semantic_freq_threshold))
        tic()
        # delete from dataset-level dicts
        synsets_to_delete = set()
        for synset in dataset_freqs:
            if dataset_freqs[synset] < self.semantic_freq_threshold:
                synsets_to_delete.add(synset)
        toc("Dataset-level frequency filtering")
        # if forcing reference, we can override the freq threshold for these synsets
        if force_reference:
            orig_num = len(synsets_to_delete)
            synsets_to_delete = [x for x in synsets_to_delete if x not in self.reference_synsets]
            info("Limiting synsets-to-delete from {} to {} due to forcing to reference synset set".format(orig_num, len(synsets_to_delete)))
        if not synsets_to_delete:
            return  freq_dict_list, dataset_freqs
        info("Will remove {}/{} synsets due to a freq threshold of {}".format(len(synsets_to_delete), len(dataset_freqs), self.semantic_freq_threshold))
        tic()
        # delete
        for synset in synsets_to_delete:
            del dataset_freqs[synset]
            for doc_dict in freq_dict_list:
                if synset in doc_dict:
                    del doc_dict[synset]
        toc("Document-level frequency filtering")
        info("Synset frequency filtering resulted in {} synsets.".format(len(dataset_freqs)))
        return  freq_dict_list, dataset_freqs

    # merge list of document-wise frequency dicts
    # to a single, dataset-wise frequency dict
    def doc_to_dset_freqs(self, freq_dict_list, force_reference = False):
        dataset_freqs = {}
        for doc_dict in freq_dict_list:
            for synset, freq in doc_dict.items():
                if synset not in dataset_freqs:
                    dataset_freqs[synset] = 0
                dataset_freqs[synset] += freq
        # frequency filtering, if defined
        if self.semantic_freq_threshold:
            freq_dict_list, dataset_freqs = self.apply_freq_filtering(freq_dict_list, dataset_freqs, force_reference)

        # complete document-level freqs with zeros for dataset-level synsets missing in the document level
        #for d, doc_dict in enumerate(freq_dict_list):
        #    for synset in [s for s in dataset_freqs if s not in doc_dict]:
        #        freq_dict_list[d][synset] = 0
        return dataset_freqs, freq_dict_list


    # tf-idf computation
    def compute_tfidf_weights(self, current_synset_freqs, dataset_freqs, force_reference=False):
        # compute tf-idf
        tic()
        tfidf_freqs = []
        for doc_dict in range(len(current_synset_freqs)):
            ddict = {}
            for synset in current_synset_freqs[doc_dict]:
                if dataset_freqs[synset] > 0:
                    ddict[synset] = current_synset_freqs[doc_dict][synset] / dataset_freqs[synset]
                else:
                    ddict[synset] = 0

            tfidf_freqs.append(ddict)
        self.synset_tfidf_freqs.append(tfidf_freqs)
        toc("tf-idf computation")

    # map a single dataset portion
    def map_dset(self, dset_words_pos, store_reference_synsets = False, force_reference_synsets = False):
        if force_reference_synsets:
            # restrict discoverable synsets to a predefined selection
            # used for the test set where not encountered synsets are unusable
            info("Restricting synsets to the reference synset set of {} entries.".format(len(self.reference_synsets)))

        current_synset_freqs = []
        tic()
        for wl, word_info_list in enumerate(dset_words_pos):
            debug("Semantic processing for document {}/{}".format(wl+1, len(dset_words_pos)))
            doc_freqs = {}
            for w, word_info in enumerate(word_info_list):
                synset, doc_freqs = self.get_synset(word_info, doc_freqs, force_reference_synsets)
                if not synset: continue
                self.process_synset(synset)
            if not doc_freqs:
                warning("No synset information extracted for document {}/{}".format(wl+1, len(dset_words_pos)))
            current_synset_freqs.append(doc_freqs)
        toc("Document-level mapping and frequency computation")
        # merge to dataset-wise synset frequencies
        tic()
        dataset_freqs, current_synset_freqs = self.doc_to_dset_freqs(current_synset_freqs, force_reference = force_reference_synsets)
        self.dataset_freqs.append(dataset_freqs)
        self.synset_freqs.append(current_synset_freqs)
        toc("Dataset-level frequency computation")

        if store_reference_synsets:
            self.reference_synsets = set((dataset_freqs.keys()))

        # compute tfidf
        if self.semantic_weights == "tfidf":
            self.compute_tfidf_weights(current_synset_freqs, dataset_freqs, force_reference=force_reference_synsets)


    # apply disambiguation to choose a single semantic unit from a collection of such
    def disambiguate(self, synsets, word_information):
        if self.disambiguation == "first":
            res =  synsets[0]._name
        elif self.disambiguation == 'POS':
            # part-of-speech filtering
            word, word_pos = word_information
            pass
        elif self.disambiguation == 'embedding-centroid':
            # generate closest synset embeddings
            # assign to closest embedding
            pass
        elif self.disambiguation == "prior":
            # select the synset with the highest prior prob
            pass
        else:
            error("Undefined disambiguation method: " + self.disambiguation)


    def get_synset_from_context_embeddings(self, word):
        word_embedding = self.embbedings.get_embeddings(word)
        self.embeddings.get_nearest_embedding(word_embedding)

    # generate semantic embeddings from words associated with an synset
    def compute_semantic_embeddings(self):
        if exists(self.sem_embeddings_path):
            info("Loading existing semantic embeddings from {}.".format(self.sem_embeddings_path))
            with open(self.sem_embeddings_path, "rb") as f:
                self.synset_embeddings, self.reference_synsets = pickle.load(f)
                return

        info("Computing semantic embeddings, using {} embeddings of dim {}.".format(self.embedding.name, self.semantic_embedding_dim))
        self.synset_embeddings = np.ndarray((0, self.embedding.embedding_dim), np.float32)
        for s, synset in enumerate(self.reference_synsets):
            # get the embeddings for the words of the synset
            words = self.semantic_context[synset]
            debug("Reference synset {}/{}: {}, context words: {}".format(s+1, len(self.reference_synsets), synset, len(words)))
            # get words not in cache
            word_embeddings = self.embedding.get_embeddings(words)

            # aggregate
            if self.semantic_embedding_aggr == "avg":
                embedding = np.mean(word_embeddings.as_matrix(), axis=0)
                self.synset_embeddings = np.vstack([self.synset_embeddings, embedding])
            else:
                error("Undefined semantic embedding aggregation:{}".format(self.semantic_embedding_aggr))

        # save results
        info("Writing semantic embeddings to {}".format(self.sem_embeddings_path))
        with open(self.sem_embeddings_path, "wb") as f:
            pickle.dump([self.synset_embeddings, self.reference_synsets], f)


    # function to map words to wordnet concepts
    def map_text(self, embedding, dataset):
        self.embedding = embedding
        if self.loaded_preprocessed:
            return

        # process semantic embeddings, if applicable
        if self.disambiguation == "context-embedding":
            self.compute_semantic_embeddings()


        # process the data
        dataset_pos = dataset.get_pos(embedding.get_present_word_indexes())
        self.synset_freqs = []
        for d, dset in enumerate(dataset_pos):
            info("Extracting {} semantic information from dataset {}/{}".format(self.name, d+1, len(dataset_pos)))
            # process data within a dataset portion
            # should store reference synsets in the train portion (d==0), but only if a reference has not
            # been already defined, e.g. via semantic embedding precomputations
            store_as_reference = (d==0 and not self.reference_synsets)
            # should force mapping to the reference if a reference has been defined
            force_reference = bool(self.reference_synsets)
            self.map_dset(dset, store_as_reference, force_reference)

        # write results: word assignments, raw, dataset-wise and tf-idf weights
        info("Writing semantic assignment results to {}.".format(self.serialization_path))
        write_pickled(self.serialization_path_preprocessed, self.get_all_preprocessed())
        info("Semantic mapping completed.")

    def get_all_preprocessed(self):
        return [self.assignments, self.synset_freqs, self.dataset_freqs, self.synset_tfidf_freqs, self.reference_synsets, self.present_word_indexes]

    def get_raw_path(self):
        return None

    # function to get a synset from a word, using the wordnet api
    # and a local word cache. Updates synset frequencies as well.
    def get_synset(self, word_information, freqs, force_reference_synsets = False):
        word, _ = word_information
        if word in self.word_synset_lookup_cache:
            synset = self.word_synset_lookup_cache[word]

            if force_reference_synsets:
                if synset not in self.reference_synsets:
                    return None, freqs

            if synset not in freqs:
                freqs[synset] = 0
        else:
            # not in cache, extract

            if self.disambiguation == "embedding-context":
                # discover synset from its generated embedding and the word embedding
                synset = self.get_synset_from_context_embeddings(word)
            else:
                # look in wordnet 
                synset = self.lookup_wordnet(word_information)
            if not synset:
                return None, freqs

            if force_reference_synsets:
                if synset not in self.reference_synsets:
                    return None, freqs
            freqs[synset] = 0
            self.word_synset_lookup_cache[word] = synset

        freqs[synset] += 1
        if word not in self.assignments:
            self.assignments[word] = synset
        return synset, freqs


    def lookup_wordnet(self, word_information):
        word, _  = word_information
        synsets = wn.synsets(word)
        if synsets:
            return self.disambiguate(synsets, word_information)
        return None

    # function that applies the required processing
    # once a synset has been found in the input text
    def process_synset(self, synset):
        # spreading activation
        pass

    # function that applies post-processing for a collected synset graph
    def postprocess_synset(self, synset):
        # frequency / tf-idf filtering - do that with synset freqs
        # ingoing / outgoing graph ratio
        pass

    # get semantic vector information, wrt to the configuration
    def get_data(self, config):
        # map dicts to vectors
        if not self.dataset_freqs:
            error("Attempted to generate semantic vectors from empty containers")
        info("Getting {} semantic data.".format(self.semantic_weights))
        synset_order = sorted(self.reference_synsets)
        semantic_document_vectors = [[] for _ in range(len(self.synset_freqs)) ]

        if self.semantic_weights  == "freq":
            # get raw semantic frequencies
            for d in range(len(self.synset_freqs)):
                for doc_dict in self.synset_freqs[d]:
                    doc_vector = [doc_dict[s] if s in doc_dict else 0 for s in synset_order]
                    semantic_document_vectors[d].append(doc_vector)

        elif self.semantic_weights == "tfidf":
            # get tfidf weights
            for d in range(len(self.synset_tfidf_freqs)):
                for doc_dict in self.synset_tfidf_freqs[d]:
                    doc_vector = [doc_dict[s] if s in doc_dict else 0 for s in synset_order]
                    semantic_document_vectors[d].append(doc_vector)

        elif self.semantic_weights == "embeddings":
            if self.disambiguation != "context-embedding":
                error("Embedding information requires the context-embedding semantic disambiguation. It is {} instead.".format(self.disambiguation))
            # get raw semantic frequencies
            for d in range(len(self.synset_freqs)):
                for doc_index, doc_dict in enumerate(self.synset_freqs[d]):
                    doc_sem_embeddings = np.ndarray((0, self.semantic_embedding_dim), np.float32)
                    if not doc_dict:
                        warning("Attempting to get semantic embedding vectors of document {}/{} with no semantic mappings. Defaulting to zero vector.".format(doc_index+1, len(self.synset_freqs[d])))
                        doc_vector = np.zeros((self.semantic_embedding_dim,), np.float32)
                    else:
                        # gather semantic embeddings of all document synsets
                        for synset in doc_dict:
                            synset_index = synset_order.index(synset)
                            doc_sem_embeddings = np.vstack([doc_sem_embeddings, self.synset_embeddings[synset_index, :]])
                        # aggregate them
                        if self.semantic_embedding_aggr == "avg":
                            doc_vector = np.mean(doc_sem_embeddings, axis=0)
                    semantic_document_vectors[d].append(doc_vector)
        else:
            error("Unimplemented semantic vector method: {}.".format(self.semantic_weights))

        return semantic_document_vectors


class GoogleKnowledgeGraph:
    pass
class PPDB:
    # ppdb reading code:
    # https://github.com/erickrf/ppdb
    pass
