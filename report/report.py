from component.component import Component
import numpy as np
from utils import to_namedtuple, debug, align_index
from bundle.datausages import *
from bundle.datatypes import *
import json

class Report(Component):
    """
    Task-specific reporting class
    """
    def load_model_from_disk(self):
        return True
    def set_component_outputs(self):
        # mark report existence
        self.data_pool.add_explicit_output(self.name)

class MultistageClassificationReport(Report):
    name = "multistageclassif"
    def __init__(self, config):
        self.config = config
        self.params = to_namedtuple(self.config.params, "params")
    
    def produce_outputs(self):
        # get reference data by chain name output
        self.label_mapping = []
        for mapping in self.params.label_mappings:
            # read json
            if type(mapping) is str and mapping.endswith(".json"):
                with open(mapping) as f:
                    mapping = json.load(f)
            mapping_dict = {ix: val for (ix, val) in enumerate(mapping)}
            self.label_mapping.append(mapping_dict)

        datapack = [x for x in self.data_pool.data if x.chain == self.params.data_chain][0]
        predictions, tagged_idx = [], []
        for i, chain_name in enumerate(self.params.pred_chains):
            # predictions
            chain_preds = [x for x in self.data_pool.data if x.chain == chain_name][0]
            predictions.append(chain_preds)

            # get tagged index
            idx_tag_name = self.params.idx_tags[i]
            if idx_tag_name is None:
                continue
            # get data with indices
            idx_data = [x for x in self.data_pool.data if type(x.data) == DummyData and x.has_usage(Indices, allow_superclasses=False)]
            # get data with indices with the desired tag
            idx_data = [x for x in idx_data if idx_tag_name in x.get_usage(Indices,allow_superclasses=False).tags][0]
            idx_data = idx_data.get_usage(Indices, allow_superclasses=False)
            idx = idx_data.get_tag_instances(idx_tag_name)
            tagged_idx.append(idx)

        # for text data, keep just the words
        if type(datapack.data) == Text:
            data = [x["words"] for x in datapack.data.instances]
        res = []

        # iterate over input instances (prior to ngramizing)
        ngram_tags = sorted([x for x in datapack.usages[0].tags if x.startswith("ngram_inst")])
        for n, ngram_tag in enumerate(ngram_tags):
            original_instance_idx = datapack.usages[0].get_tag_instances(ngram_tag)
            obj = {"instance": ngram_tag, "data": [data[i] for i in original_instance_idx], "predictions": []}
            preds_obj = {}
            for i in range(len(predictions)):
                debug(f"Creating report for instance {n+1}/{len(ngram_tags)}, prediction {i+1}/{len(predictions)}")
                preds_obj = {"step": i}
                # get all predictions & tresholding results for them
                all_preds = predictions[i].data.instances
                all_surviving_idx = tagged_idx[i]

                # get *thresholded* predictions relevant to the current original instance
                surv_instance_idx = np.intersect1d(original_instance_idx, all_surviving_idx)
                preds_obj["survivors"] = [i for (i, idx) in enumerate(original_instance_idx) if idx in surv_instance_idx]

                # sort descending, get top k
                # for completeness, get the topK preds for all instances, surviving or not
                top_k_preds, top_k_predicted_classes = self.get_topK_preds(all_preds[original_instance_idx], self.label_mapping[i])

                # write stuff in the object dict:
                # (do not write all instance predictions)
                # preds_obj[self.params.pred_chains[i]] = all_instance_preds.tolist()
                # (do not write all predictions thresholding)
                # surviving = np.zeros(len(all_preds))
                # surviving[surv_instance_idx] = 1
                # preds_obj["thresholded"] = surviving.tolist()

                # write top-k
                preds_obj["topk_preds"] = top_k_preds
                preds_obj["topk_classes"] = top_k_predicted_classes

                # eliminate instance indexes that did not survive
                # and align them to the thresholded orig. instance container
                original_instance_idx = align_index(original_instance_idx, [i in all_surviving_idx for i in range(len(all_preds))], mask_shows_deletion=False)

                # append the predictions for the instance
                obj["predictions"].append(preds_obj)

            res.append(obj)
        # add input configuration data
        pars = [dp for dp in self.data_pool.data if type(dp.data) == Dictionary][0]
        self.result = {"results": res, "params": pars.data.instances}

    def get_topK_preds(self, predictions, label_mapping):
        """
        Return topK predictions and predicted classes from a predictions matrix and index label mapping dict
        """
        if predictions.size == 0:
            top_k_preds = []
            top_k_predicted_classes = []
        else:
            # argsort the column prediction probas descending, get top k
            top_k_idxs = np.argsort(predictions, axis=1)[:,::-1][:, :self.params.top_k]
            # make a reordered probs container
            top_k_preds = [row[top_k_idxs[row_idx]].tolist() for row_idx, row in enumerate(predictions)]
            # take the classses corresponding to the argsorted indexes probs
            top_k_predicted_classes = [[label_mapping[ix] for ix in idxs] for idxs in top_k_idxs]
        return top_k_preds, top_k_predicted_classes



    def set_component_outputs(self):
        super().set_component_outputs()
        res = DataPack(Dictionary(self.result))
        res.id = f"{self.name}_report"
        self.data_pool.add_data_packs([res], self.name)

    def get_component_inputs(self):
        # orig_data = self.data_pool.get_current_inputs()[0]
        # classification_results = self.data_pool.current_inputs()[1:] 
        pass
