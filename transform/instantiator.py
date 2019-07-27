from transform.gmm import GMMClustering
from transform.kmeans import KMeansClustering
from transform.lda import LDA
from transform.pca import PCA
from transform.lida import LiDA
from transform.lsa import LSA
from utils import error


class Instantiator:
    component_name = "transform"
    avail = [LSA, KMeansClustering, GMMClustering, LiDA, LDA, PCA]

    def create(config):
        name = config.transform.name
        for tra in Instantiator.avail:
            if tra.base_name == name:
                return tra(config)
        # any unknown name is assumed to be pretrained embeddings
        error("Undefined feature transformation: {}, available ones are {}".format(name, [tra.base_name for tra in Instantiator.avail]))

