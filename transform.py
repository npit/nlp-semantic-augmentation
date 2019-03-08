from utils import error, info, shapes_list, write_pickled, debug
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from serializable import Serializable

"""Module for feature transformation methods

This module provides methods that transform existing representations into others.
"""


class Transform(Serializable):
    """Abstract transform class

    """
    name = None
    dimension = None
    dir_name = "transform"
    do_reinitialize = None
    transformer = None
    process_func_train = None
    process_func_test = None
    is_supervised = None
    term_components = None

    @staticmethod
    def create(representation):
        config = representation.config
        name = config.transform.name
        if name == LSA.base_name:
            return LSA(representation)
        if name == KMeansClustering.base_name:
            return KMeansClustering(representation)
        if name == GMMClustering.base_name:
            return GMMClustering(representation)
        if name == LiDA.base_name:
            return LiDA(representation)
        if name == LDA.base_name:
            return LDA(representation)
        # any unknown name is assumed to be pretrained embeddings
        error("Undefined feature transformation: {}, available ones are {}".format(name, Transform.get_available()))


    @staticmethod
    def get_available():
        return [cls.base_name for cls in Transform.__subclasses__()]

    def __init__(self, representation):
        config = representation.config
        self.do_reinitialize = False
        self.is_supervised = False
        self.config = config
        self.dimension = config.transform.dimension
        if representation.dimension is not None:
            self.initialize(representation)
            self.acquire_data()
        else:
            # None dimension (typical for uncomputed bags) -- no point to load
            info("Will not attempt to load data of a <None> representation dimension transform.")
            self.name = "none_dim_repr"
            Serializable.__init__(self, self.dir_name)
            self.do_reinitialize = True
            return

    def initialize(self, representation):
        self.name = "{}_{}_{}".format(representation.name, self.base_name, self.dimension)
        Serializable.__init__(self, self.dir_name)
        self.set_serialization_params()

    def get_dimension(self):
        return self.dimension

    def get_name(self):
        """Composite name getter"""
        return "{}_{}".format(self.name, self.dimension)

    def check_compatibility(self, dataset, repres):
        if repres.get_dimension() < self.dimension:
            error("Got transform dimension of {} but representation of {}.".format(self.dimension, repres.get_dimension()))
        return True

    def compute(self, repres, dataset):
        """Apply transform on input features"""
        if self.loaded():
            info("Skipping {} computation due to transform data already loaded.".format(self.name))
            return
        if not repres.need_load_transform():
            info("Skipping {} computation due to encompassing representations are already loaded. aggregated or finalized data already loaded.".format(self.name))
            return
        if self.do_reinitialize:
            info("Reinitializing transform parameters from the loaded representation config.")
            self.initialize(repres)
            self.acquire_data()
            if self.loaded():
                info("Loaded existing transformed data after reinitializion, exiting.")
                return

        # sanity checks
        self.check_compatibility(dataset, repres)

        # compute
        num_chunks = len(repres.dataset_vectors)
        info("Applying {} {}-dimensional transform on the raw representation.".format(self.base_name, self.dimension))
        self.term_components = []
        for dset_idx, dset in enumerate(repres.dataset_vectors):
            # replace non-reduced vectors, to save memory
            if dset_idx == 0:
                info("Transforming training input data shape {}/{}: {}".format(dset_idx + 1, num_chunks, dset.shape))
                if self.is_supervised:
                    ground_truth = dataset.get_targets()[dset_idx]
                    ground_truth = repres.match_targets_to_instances(dset_idx, ground_truth)
                    repres.dataset_vectors[dset_idx] = self.process_func_train(dset, ground_truth)
                else:
                    repres.dataset_vectors[dset_idx] = self.process_func_train(dset)
            else:
                info("Transforming test input data shape {}/{}: {}".format(dset_idx + 1, num_chunks, dset.shape))
                repres.dataset_vectors[dset_idx] = self.process_func_test(dset)
            self.verify_dimensionality(repres.dataset_vectors[dset_idx])
            self.term_components.append(self.get_term_representations())

        repres.dimension = self.dimension
        info("Output shapes (train/test): {}, {}".format(*shapes_list(repres.dataset_vectors)))
        write_pickled(self.serialization_path_preprocessed, repres.get_all_preprocessed())
        pass

    def get_raw_path(self):
        return None

    def get_all_preprocessed(self):
        return self.repr_data

    def handle_preprocessed(self, data):
        self.repr_data = data

    def get_term_representations(self):
        """Return term-based, rather than document-based representations
        """
        return self.transformer.components_

    def verify_dimensionality(self, data):
        """Checks if the projected data dimension matches the one prescribed
        """
        data_dim = data.shape[-1]
        if data_dim != self.dimension:
            error("{} result dimension {} does not match the prescribed input dimension {}".format(self.name, data_dim, self.dimension))


class LSA(Transform):
    """Latent Semantic Analysis decomposition.

    Based on the truncated SVD implementation of sklearn.
    """
    base_name = "lsa"

    def __init__(self, representation):
        """LSA constructor"""
        Transform.__init__(self, representation)
        self.transformer = TruncatedSVD(self.dimension)
        self.process_func_train = self.transformer.fit_transform
        self.process_func_test = self.transformer.transform


class GMMClustering(Transform):
    """Gausian Mixture Model clustering

    Uses the Gaussian mixture implementation of sklearn.
    """
    base_name = "gmm"

    def __init__(self, representation):
        """GMM-clustering constructor"""
        Transform.__init__(self, representation)
        self.transformer = GaussianMixture(self.dimension)
        self.process_func_train = self.fit_predict_proba
        self.process_func_test = self.transformer.predict_proba

    def fit_predict_proba(self, data):
        self.transformer = self.transformer.fit(data)
        return self.process_func_test(data)

    def get_term_representations(self):
        """Return term-based, rather than document-based representations
        """
        return self.transformer.means_


class KMeansClustering(Transform):
    """KMeans clustering

    Uses the KMeans implementation of sklearn.
    """
    base_name = "kmeans"

    def __init__(self, representation):
        """Kmeans-clustering constructor"""
        Transform.__init__(self, representation)
        self.transformer = KMeans(self.dimension)
        self.process_func_train = self.fit
        self.process_func_test = self.do_transform

    def fit(self, data):
        self.transformer = self.transformer.fit(data)
        return self.do_transform(data)

    def do_transform(self, data):
        res = self.transformer.transform(data)
        return res

    def get_term_representations(self):
        """Return term-based, rather than document-based representations
        """
        return self.transformer.cluster_centers_


class LiDA(Transform):
    """Linear Discriminant Analysis transformation

    Uses the LiDA implementation of sklearn.
    """
    base_name = "lida"

    def __init__(self, representation):
        """LiDA constructor"""
        Transform.__init__(self, representation)
        self.transformer = LinearDiscriminantAnalysis(n_components=self.dimension)
        self.is_supervised = True
        self.process_func_train = self.transformer.fit_transform
        self.process_func_test = self.transformer.transform

    def check_compatibility(self, dataset, repres):
        if dataset.is_multilabel():
            error("{} transform is not compatible with multi-label data.".format(self.base_name))
        if not (self.dimension < dataset.get_num_labels() - 1):
            error("The {} projection dimension ({}) needs to be less than the dataset classes minus one ({} -1 = {})".\
                  format(self.base_name, self.dimension, dataset.get_num_labels(), dataset.get_num_labels() - 1))

    def get_term_representations(self):
        """Return term-based, rather than document-based representations
        """
        return self.transformer.means_


class LDA(Transform):
    """Latent Dirichlet Allocation transformation

    Uses the LDA implementation of sklearn.
    """
    base_name = "lda"

    def __init__(self, config):
        self.config = config
        Transform.__init__(self, config)
        self.transformer = LatentDirichletAllocation(n_components=self.dimension, random_state=self.config.get_seed())
        self.process_func_train = self.transformer.fit_transform
        self.process_func_test = self.transformer.transform
