from transform.gmm import GMMClustering
from transform.kmeans import KMeansClustering
from transform.lda import LDA
from transform.lida import LiDA
from transform.lsa import LSA
from transform.pca import PCA
from utils import error


class Instantiator:
    component_name = "transform"
    avail = [LSA, KMeansClustering, GMMClustering, LiDA, LDA, PCA]

    def create(config):
        name = config.name
        for tra in Instantiator.avail:
            if tra.base_name == name:
                return tra(config)
        # any unknown name is assumed to be pretrained embeddings
        error(f"Undefined feature transformation: {name}, available ones are {[tra.base_name for tra in Instantiator.avail]}")
