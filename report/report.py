from component.component import Component
import numpy as np
from utils import to_namedtuple, debug, align_index, error, as_list, tictoc
from bundle.datausages import *
from bundle.datatypes import *
import json

class Report(Component):
    """
    Task-specific reporting class
    """
    component_name = "report"

    def load_model_from_disk(self):
        return True
    def set_component_outputs(self):
        # mark report existence
        self.data_pool.add_explicit_output(self.name)


class IndexMapper:
    """
    Class to convert between indexes
    """
    def __init__(self, size, list_of_indexes):
        """
        size: Size of the original container 
        list_of_indexes: each contains indices to elements of a container.
                        Implicitly defines a new container with the elements it points to, where the next index list refers to.
        """
        self.size = size
        self.indexes = list_of_indexes

    def index_survives(self, idx, target_level=-1):
        return self.convert_index(idx, target_level=target_level) is not None

    def convert_index_to_last_container(self, idx):
        """Converts input index to indexes of the last tangible container (to which the last index-list refers to)"""
        lvl =  len(self.indexes) - 2
        return self.convert_index(idx, target_level=lvl)

    def convert_index(self, idx, source_level=-1, target_level=None):
        """Converts index idx that resides on index_list of level source_level to the corresponding one
        at target_level"""
        error(f"Invalid index {idx}", idx < 0)
        if source_level == -1:
            # starts from the original index pool
            if idx >= self.size:
                error(f"Invalid index {idx} wrt. original size: {self.size}")
            pass
        else:
            # make sure it exists in the specified level
            if idx not in self.indexes[source_level]:
                error(f"Requested conversion of idx {idx}: lvl {source_level}->{target_level}, but it's not in the source: {self.indexes[source_level]}")
        if target_level is None:
            # by default produce indexes in the final level
            target_level = len(self.indexes) - 1

        if target_level == source_level:
            return np.asarray([idx], np.int64)

        # compute
        curr_value = idx
        curr_lvl = source_level
        while True:
            if target_level > curr_lvl:
                # next level is ahead
                curr_lvl += 1
                if idx not in self.indexes[curr_lvl]:
                    # index did not survive past this step
                    return None
                curr_value = np.where(self.indexes[curr_lvl] == curr_value)[0]
            elif target_level < curr_lvl:
                # next level is behind
                curr_lvl -= 1
                # slice; it's guaranteed to exist
                curr_value = self.indexes[curr_lvl][curr_value]
            else:
                return curr_value

class MultistageClassificationReport(Report):
    name = "multistageclassif"
    def __init__(self, config):
        self.config = config
        self.params = self.config.params

        if any(self.params[k] is None for k in "data_chain pred_chains idx_tags".split()):
            error("Need report params for keys: data_chain, pred_chains, idx_tags")

        stage_data_keys = "pred_chains idx_tags".split()
        lens = [len(self.params[k]) for k in stage_data_keys]
        if lens.count(lens[0]) != len(lens):
            error(f"Need same number of entries for stage data keys:{stage_data_keys} -- got {[self.params[k] for k in stage_data_keys]}")

        self.params["only_report_labels"] = [None] * len(self.params["pred_chains"])

        if self.params["debug"] is None:
            self.params["debug"] = False
        if self.params["only_report_labels"] is None:
            self.params["only_report_labels"] = [None] * len(self.params["pred_chains"])

        self.params = to_namedtuple(self.config.params, "params")
    
    def produce_outputs(self):
        # get input configuration data
        self.topk = None
        self.messages = []
        self.input_parameters_dict = [dp for dp in self.data_pool.data if type(dp.data) == Dictionary][0].data.instances
        self.input_parameters = to_namedtuple(self.input_parameters_dict, "input_parameters")

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

        # get final scores
        final_preds = predictions[len(predictions)-1].data.instances
        final_surv_idx = tagged_idx[len(predictions)-1]

        curr_surv_idx = final_surv_idx
        # find which words the survivors belong to
        for idx in reversed(tagged_idx[:-1]):
            curr_surv_idx = idx[curr_surv_idx]

        res = []
        # contextualize wrt. each instance (specified by the ngram tag)
        num_all_ngrams = len(predictions[0].data.instances)
        num_steps = len(predictions)
        index_mapper = IndexMapper(num_all_ngrams, tagged_idx)

        ngram_tags = sorted([x for x in datapack.usages[0].tags if x.startswith("ngram_inst")])
        with tictoc("Classification report building", announce=False):
            for n, ngram_tag in enumerate(ngram_tags):
                # indexes of the tokens for the current instance
                # to the entire data container
                original_instance_ix_data = datapack.usages[0].get_tag_instances(ngram_tag)
                inst_obj = {"instance": n, "data": [data[i] for i in original_instance_ix_data], "detailed_preds": []}

                for local_word_idx, ix in enumerate(original_instance_ix_data):
                    word_obj = {"word": data[ix], "word_idx": int(local_word_idx), "word_preds": []}
                    # for each step
                    for step_idx in range(num_steps):
                        preds = predictions[step_idx].data.instances
                        step_name = self.params.pred_chains[step_idx]
                        step_obj = {"name": step_name, "step_index": step_idx}

                        if step_idx == 0 or index_mapper.index_survives(ix, target_level=step_idx):
                            # we want the position of in the pred. container previous to the step
                            surv_idx = index_mapper.convert_index(ix, target_level=step_idx-1)
                            step_preds = preds[surv_idx, :]
                            scores, classes = self.get_topK_preds(step_preds, self.label_mapping[step_idx], self.params.only_report_labels[step_idx])
                            step_obj["step_preds"] = {c:s for (c,s) in zip(classes[0], scores[0])}
                        else:
                            scores, classes = [], []
                            step_obj["step_preds"] = {}
                        word_obj["word_preds"].append(step_obj)
                        # 
                        if step_idx == num_steps - 1:
                            word_obj["overall_preds"] = step_obj["step_preds"]

                    inst_obj["detailed_preds"].append(word_obj)
                res.append(inst_obj)

        self.result = {"results": res, "input_params": self.input_parameters_dict, "messages": self.messages}

    def align_to_original_index(self, idx_progression, original_idx):
        """
        Aligns a collection of self-applying indexes to the original collection indexes
        index_progression (list): List of index collections in temporal order
        original_idx (list): Original index collection
        """
        # set index in in reversed order and append the original
        current = idx_progression[-1]
        if not current:
            return []
        # get progression but for the last step
        all_progression = reversed([original_idx] + idx_progression[:-1])
        for active_idx in all_progression:
            if not active_idx:
                return []
            # align
            current = [active_idx[i] for i in current]
        return current

    def get_topK_preds(self, predictions, label_mapping, only_report_labels):
        """
        Return topK predictions and predicted classes from a predictions matrix and index label mapping dict
        """
        # get top k from input / static parameters, or revert to default
        if self.topk is None:
            try:
                self.topk = self.input_parameters.top_k
            except KeyError:
                self.topk = self.params.top_k
            except AttributeError:
                self.topk = 5
                self.messages.append(f"Defaulted to top_k of {self.topk}")

        if predictions.size == 0:
            top_k_preds = []
            top_k_predicted_classes = []
        else:
            # argsort the column prediction probas descending, get top k
            top_k_idxs = np.argsort(predictions, axis=1)[:,::-1][:, :self.topk]
            # make a reordered probs container
            top_k_preds = [row[top_k_idxs[row_idx]].tolist() for row_idx, row in enumerate(predictions)]
            # take the classses corresponding to the argsorted indexes probs
            top_k_predicted_classes = [[label_mapping[ix] for ix in idxs] for idxs in top_k_idxs]
        if only_report_labels is not None:
            only_report_labels = as_list(only_report_labels)
            for i in range(len(top_k_predicted_classes)):
                retained_label_idxs = []
                for lbl in only_report_labels:
                    if lbl not in top_k_predicted_classes[i]:
                        error(f"Requested to only report label {lbl} but top {self.topk} predicted labels are {top_k_predicted_classes[i]}")
                    lbl_idx = top_k_predicted_classes[i].index(lbl)
                    retained_label_idxs.append(lbl_idx)
                top_k_predicted_classes[i] = [top_k_predicted_classes[i][l] for l in retained_label_idxs]
                top_k_preds[i] = [top_k_preds[i][l] for l in retained_label_idxs]
        return top_k_preds, top_k_predicted_classes

    def set_component_outputs(self):
        super().set_component_outputs()
        res = DataPack(Dictionary(self.result))
        res.id = f"{self.name}_report"
        self.data_pool.add_data_packs([res], self.name)

    def get_component_inputs(self):
        # handle in production
        pass