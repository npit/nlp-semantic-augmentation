from utils import error, info, shapes_list, write_pickled, debug
from sklearn.decomposition import TruncatedSVD
from sklearn.mixture import GaussianMixture
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

    @staticmethod
    def create(representation):
        config = representation.config
        name = config.transform.name
        if name == LSA.base_name:
            return LSA(representation)
        if name == Clustering.base_name:
            return Clustering(representation)
        # any unknown name is assumed to be pretrained embeddings
        error("Undefined feature transformation: {}".format(name))

    def __init__(self, representation):
        config = representation.config
        self.do_reinitialize = False
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

    def compute(self, repres):
        """Apply transform on input features"""
        if self.loaded():
            debug("Skipping {} computation due to data already loaded.".format(self.name))
            return
        if self.do_reinitialize:
            info("Reinitializing transform parameters from the loaded representation config.")
            self.initialize(repres)
            self.acquire_data()
            if self.loaded():
                info("Loaded existing transformed data after reinitializion, exiting.")
                return

        repres = self.apply_transform(repres)
        repres.dimension = self.dimension
        info("Output shapes (train/test): {}, {}".format(*shapes_list(repres.dataset_vectors)))
        write_pickled(self.serialization_path_preprocessed, repres.get_all_preprocessed())
        pass

    def apply_transform(self, repres):
        error("Attempted to invoke apply_transform from abstract transform object, using a {} derived object.".format(self.name))

    def get_raw_path(self):
        return None

    def get_all_preprocessed(self):
        return self.repr_data

    def handle_preprocessed(self, data):
        self.repr_data = data


class LSA(Transform):
    """Latent Semantic Analysis decomposition.

    Based on the truncated SVD implementation of sklearn.
    """
    base_name = "LSA"
    do_reinitialize = None

    def __init__(self, representation):
        """LSA constructor"""
        Transform.__init__(self, representation)

    def apply_transform(self, repres):
        self.tsvd = TruncatedSVD(self.dimension)
        num_chunks = len(repres.dataset_vectors)
        info("{} computation".format(self.name))
        for dset_idx, dset in enumerate(repres.dataset_vectors):
            info("Data chunk {}/{}".format(dset_idx + 1, num_chunks))
            # replace non-reduced vectors, to save memory
            if dset_idx == 0:
                info("Computing {} transform on input data shape {}/{}: {}".format(self.name, dset_idx + 1, num_chunks, dset.shape))
                repres.dataset_vectors[dset_idx] = self.tsvd.fit_transform(dset)
            else:
                info("Applying {} transform on input data shape {}/{}: {}".format(self.name, dset_idx + 1, num_chunks, dset.shape))
                repres.dataset_vectors[dset_idx] = self.tsvd.transform(dset)
        return repres


class Clustering(Transform):
    """Clustering.

    Uses the Gaussian mixture implementation of sklearn.
    """
    base_name = "clustering"

    def apply_transform(self, repres):
        self.gmm = GaussianMixture(self.dimension)
        num_chunks = len(repres.dataset_vectors)
        info("{} computation".format(self.name))
        for dset_idx, dset in enumerate(repres.dataset_vectors):
            info("Data chunk {}/{}".format(dset_idx + 1, num_chunks))
            # replace non-reduced vectors, to save memory
            if dset_idx == 0:
                info("Computing {} transform on input data shape {}/{}: {}".format(self.name, dset_idx + 1, num_chunks, dset.shape))
                repres.dataset_vectors[dset_idx] = self.gmm.fit_predict(dset)
            else:
                info("Applying {} transform on input data shape {}/{}: {}".format(self.name, dset_idx + 1, num_chunks, dset.shape))
                repres.dataset_vectors[dset_idx] = self.gmm.predict(dset)
        return repres
