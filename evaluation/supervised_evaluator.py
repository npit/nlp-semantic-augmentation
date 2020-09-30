from bundle import DataPool
from bundle.datatypes import *
from bundle.datausages import *

from sklearn import metrics

class SupervisedEvaluator(Evaluator):
    """Evaluator for supervised tasks"""

    consumes = [Numeric.name, GroundTruth.name]
    available_measures = ("silhouette", "rouge")
    self.measure_funcs = {"rouge": SupervisedEvaluator.compute_rouge}

    def __init__(self, config):
        self.config = config
        super().__init__(config)


    def get_component_inputs(self):
        """Get inputs for unsupervised evaluation"""
        # anything but ground truth
        matches = self.data_pool.request_data(None, GroundTruth, usage_matching="subset", client=self.name)
        self.data = matches.data
        self.indices = matches.get_usage(Indices.name)

    def compute_rouge(self, data):
        """Compute rouge"""
        gt, preds = data
        preds = np.argmax(preds, axis=1)
        maxlen = preds.shape[-1]
        evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l', 'rouge-w'], max_n=4, limit_length=True, length_limit=maxlen, 
                length_limit_type='words',
                apply_avg=True,
                alpha=0.5, # Default F1_score
                weight_factor=1.2, stemming=True)
        scores = evaluator.get_scores(gt, preds)
