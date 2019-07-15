from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from transform.transform import Transform
from utils import error


class LiDA(Transform):
    """Linear Discriminant Analysis transformation

    Uses the LiDA implementation of sklearn.
    """
    base_name = "lida"

    def __init__(self, representation):
        """LiDA constructor"""
        Transform.__init__(self, representation)
        # supervised
        self.transformer = LinearDiscriminantAnalysis(n_components=self.dimension)
        self.is_supervised = True
        self.process_func_train = self.transformer.fit_transform
        self.process_func_test = self.transformer.transform

    def check_compatibility(self, dataset, repres):
        if dataset.is_multilabel():
            error("{} transform is not compatible with multi-label data.".format(self.base_name))
        if not (self.dimension < dataset.get_num_labels() - 1):
            error("The {} projection dimension ({}) needs to be less than the dataset classes minus one ({} -1 = {})". \
                  format(self.base_name, self.dimension, dataset.get_num_labels(), dataset.get_num_labels() - 1))

    def get_term_representations(self):
        """Return term-based, rather than document-based representations
        """
        return self.transformer.means_

