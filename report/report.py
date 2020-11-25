from component.component import Component
import numpy as np
from utils import to_namedtuple, debug, align_index, error, as_list
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
        if "debug" not in self.config.params:
            self.config.params["debug"] = False
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

        # iterate over input instances (prior to ngramizing)
        ngram_tags = sorted([x for x in datapack.usages[0].tags if x.startswith("ngram_inst")])
        for n, ngram_tag in enumerate(ngram_tags):
            # indexes of the tokens for the current istance to the entire data container
            original_instance_ix_data = datapack.usages[0].get_tag_instances(ngram_tag)

            # indexes of the current tokens for the current istance to the prediction container
            # (the first prediction batch is on the entire data)
            instance_ix_preds = original_instance_ix_data.copy()
            # indexes of the current tokens for the current istance to the entire data container
            instance_ix_data = original_instance_ix_data.copy()

            # indexes (size == num curr. curr. predictions) mapping each pred. row
            # to each token in the data container
            prediction_ix_data = None

            obj = {"instance": n, "data": [data[i] for i in original_instance_ix_data], "predictions": []}
            preds_obj = {}
            # iterate over prediction steps
            for i in range(len(predictions)):
                debug(f"Creating report for instance {n+1}/{len(ngram_tags)}, prediction {i+1}/{len(predictions)}")
                preds_obj = {"step": i}
                # all prediction scores
                step_preds = predictions[i].data.instances
                if prediction_ix_data is None:
                    # this is the first step -- size should equal
                    prediction_ix_data = np.arange(len(step_preds))
                    error("Mismatch in number of predictions and data!", len(prediction_ix_data) != len(data))

                # filter out survivors
                # local index of entries surviving the thresholding
                survivor_ix_preds = tagged_idx[i]
                survivor_ix_data = prediction_ix_data[survivor_ix_preds]

                # filter out to current instance
                # (both indexes refer to the preds. container):
                instance_survivor_ix_preds = np.intersect1d(instance_ix_preds, survivor_ix_preds)
                # (both indexes refer to the data. container):
                instance_survivor_ix_data = np.intersect1d(survivor_ix_data, instance_ix_data)

                preds_obj["survivor_words"] = [data[i] for i in instance_survivor_ix_data]

                ix = [np.where(original_instance_ix_data == d)[0] for d in instance_survivor_ix_data]
                ix = np.concatenate(ix) if ix else np.asarray([],dtype=np.int64)
                preds_obj["survivor_word_instance_index"] = ix.tolist()

                # extra info for verification debugging
                ########################################
                if i == 0:
                    th = self.input_parameters_dict["binary_threshold"]
                    if np.any(step_preds[survivor_ix_preds,1] < th):
                        print("bin Threshold!")
                else:
                    th = self.input_parameters_dict["multiclass_threshold"]
                    l1 = len(np.unique(np.where(step_preds[survivor_ix_preds,:] >= th)[0]))
                    l2 = len(survivor_ix_preds)
                    if l1 != l2:
                        print("mc Threshold!")
                ds = [data[i] for i in instance_survivor_ix_data]
                dall = [data[i] for i in original_instance_ix_data]
                if any(x not in dall for x in ds):
                    print("words")

                if self.params.debug or ("debug" in self.input_parameters_dict and self.input_parameters_dict["debug"]):
                    preds_obj["survivor_inst_idx_wrt_preds"] = instance_survivor_ix_preds.tolist()
                    preds_obj["survivor_inst_idx_wrt_data"] = instance_survivor_ix_data.tolist()
                    preds_obj["survivor_idx_wrt_preds"] = survivor_ix_preds.tolist()
                    preds_obj["survivor_idx_wrt_data"] = survivor_ix_data.tolist()

                    preds_obj["prediction_ix_wrt_data"] = prediction_ix_data.tolist()
                    preds_obj["instance_idx_wrt_preds"] = instance_ix_preds.tolist()

                    preds_obj["topk_in_predictions"] = []
                    top_k_all, top_k_classes_all = self.get_topK_preds(step_preds, self.label_mapping[i], self.params.only_report_labels[i])
                    for (pr, cl) in zip (top_k_all, top_k_classes_all):
                        preds_obj["topk_in_predictions"].append({c: p for (c, p) in zip(cl, pr)})

                ########################################

                # sort descending, get top k
                # for completeness, get the topK preds for all instances, surviving or not
                top_k_preds, top_k_predicted_classes = self.get_topK_preds(step_preds[instance_ix_preds], self.label_mapping[i],self.params.only_report_labels[i])

                # write top-k
                preds_obj["topk_per_instance_token"] = []
                for (pr, cl) in zip (top_k_preds, top_k_predicted_classes):
                    preds_obj["topk_per_instance_token"].append({c: p for (c, p) in zip(cl, pr)})
                
                # for the next step, only survivors remain
                # predictions
                prediction_ix_data = survivor_ix_data
                # instance-related:
                # ixs to data is straightforward
                instance_ix_data = instance_survivor_ix_data
                # aligning the surviving indexes of the instance to the surviving preds container
                # via the surviving data
                ix = [np.where(prediction_ix_data == d)[0] for d in instance_survivor_ix_data]
                instance_ix_preds = np.concatenate(ix) if ix else np.asarray([],dtype=np.int64)

                # preds_obj["topk_classes"] = top_k_predicted_classes

                # append the predictions for the instance
                obj["predictions"].append(preds_obj)

            # total predictions
            obj["overall_predictions"] = []

            orig_words = [data[i] for i in instance_ix_data]
            for s, surv_idx in enumerate(instance_ix_data):
                widx = np.where(original_instance_ix_data == surv_idx)[0]
                tobj = {"original_word_index": int(widx), "original_word": orig_words[s]}
                tobj["replacements"] = {}
                for (pr, cl) in zip (top_k_preds[s], top_k_predicted_classes[s]):
                    tobj["replacements"][cl] = pr
                obj["overall_predictions"].append(tobj)

            res.append(obj)
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