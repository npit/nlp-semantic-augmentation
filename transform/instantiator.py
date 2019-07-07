from transform.gmm import GMMClustering
from transform.kmeans import KMeansClustering
from transform.lda import LDA
from transform.lida import LiDA
from transform.lsa import LSA
from utils import error


class Instantiator:
    component_name = "transform"

    def create(config):
        name = config.transform.name
        if name == LSA.base_name:
            return LSA(config)
        if name == KMeansClustering.base_name:
            return KMeansClustering(config)
        if name == GMMClustering.base_name:
            return GMMClustering(config)
        if name == LiDA.base_name:
            return LiDA(config)
        if name == LDA.base_name:
            return LDA(config)

        available = [LSA.base_name, KMeansClustering.base_name, GMMClustering.base_name, LiDA.base_name, LDA.base_name]
        # any unknown name is assumed to be pretrained embeddings
        error("Undefined feature transformation: {}, available ones are {}".format(name, available))

