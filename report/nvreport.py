from component.component import Component
import numpy as np
from utils import to_namedtuple, debug, align_index, error, as_list, tictoc
from bundle.datausages import *
from bundle.datatypes import *
import json
from report.report import Report



class NVReport(Report):
    name = "nvreport"
    def __init__(self, config):
        self.config = config
        self.params = self.config.params

        if any(self.params[k] is None for k in "data_chain pred_chains".split()):
            error("Need report params for keys: data_chain, pred_chains, idx_tags")

        self.params["only_report_labels"] = [None] * len(self.params["pred_chains"])

        if self.params["debug"] is None:
            self.params["debug"] = False
        if self.params["only_report_labels"] is None:
            self.params["only_report_labels"] = [None] * len(self.params["pred_chains"])
        # if "final_stages" not in self.params:
        #     self.params["final_stages"] = None
        if "report_if_fail" not in self.params:
            self.params["report_if_fail"] = None

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
            if type(mapping) is str:
                try:
                    with open(mapping) as f:
                        mapping = json.load(f)
                except:
                    error("Requires json labelmapping or literal list")
            mapping_dict = {ix: val for (ix, val) in enumerate(mapping)}
            self.label_mapping.append(mapping_dict)


        # thresholding
        for th in self.params.thresholds:
            if th not in self.input_parameters_dict:
                self.result =  {"results": [], "input_params": self.input_parameters_dict, "messages": [f"Threshold {th} missing from input parameters"]}
                return


        datapack = [x for x in self.data_pool.data if x.chain == self.params.data_chain][0]
        predictions, tagged_idx = [], []
        for i, chain_name in enumerate(self.params.pred_chains):
            # predictions
            chain_preds = [x for x in self.data_pool.data if x.chain == chain_name][0]
            predictions.append(chain_preds)


        # for text data, keep just the words
        if type(datapack.data) == Text:
            data = [x["words"] for x in datapack.data.instances]

        res = []
        
        predictions = [x.data.instances for x in predictions]
        num_all_ngrams = len(predictions[0])
        num_steps = len(predictions)

        # compute thresholding values
        thresholding = np.zeros((num_all_ngrams, len(self.params.thresholds)), bool)
        # for i, th in enumerate(self.params.thresholds):
        #     th_val = float(self.input_parameters_dict[th])


        thresholding[:, 0] = predictions[0][:, 1] > float(self.input_parameters_dict[self.params.thresholds[0]])
        thresholding[:, 1] = predictions[1][:, 1] > float(self.input_parameters_dict[self.params.thresholds[1]])
        thresholding[:, 2] = np.any(predictions[2] > float(self.input_parameters_dict[self.params.thresholds[2]]), axis=1)


        ngram_tags = sorted([x for x in datapack.usages[0].tags if x.startswith("ngram_inst")])

        with tictoc("Classification report building", announce=False):
            for n, ngram_tag in enumerate(ngram_tags):
                # indexes of the tokens for the current instance
                # to the entire data container
                original_instance_ix_data = datapack.usages[0].get_tag_instances(ngram_tag)
                inst_obj = {"instance": n, "data": [data[i] for i in original_instance_ix_data], "predictions": []}

                for local_word_idx, ix in enumerate(original_instance_ix_data):
                    word_obj = {"word": data[ix], "word_idx": int(local_word_idx), "overall_predictions": {}}
                    detailed = []
                    
                    # for each step
                    for step_idx in range(num_steps):
                        preds = predictions[step_idx]
                        step_name = self.params.pred_chains[step_idx]
                        step_obj = {"name": step_name, "step_index": step_idx}

                        survives = thresholding[ix, step_idx]
                        step_preds = np.expand_dims(preds[ix, :], axis=0)
                        scores, classes = self.get_topK_preds(step_preds, self.label_mapping[step_idx], self.params.only_report_labels[step_idx])
                        step_preds = {c: round(s, 4) for (c, s) in zip(classes[0], scores[0])}
                        step_obj["step_preds"] = step_preds
                        detailed.append(step_preds)

                    modified, deleted, replaced = thresholding[ix, :]
                    word_obj["overall_predictions"]["modify_prediction"] = {"modified": int(modified), "prob": detailed[0]["modify"]}

                    delete_obj = detailed[1]
                    # replaced
                    objs = []
                    for word, prob in detailed[2].items():
                        objs.append({"word": word, "prob": prob})
                    replace_obj = objs

                    if modified:
                        if deleted:
                            # deleted
                            word_obj["overall_predictions"]["delete_prediction"] = delete_obj
                        elif replaced:
                            word_obj["overall_predictions"]["replace_prediction"] = replace_obj

                    if not self.omit_detailed_results():
                        word_obj["detailed_predictions"] = {"delete_prediction": delete_obj, "replace_prediction": replace_obj}

                    inst_obj["predictions"].append(word_obj)
                res.append(inst_obj)

        self.result = {"results": res, "input_params": self.input_parameters_dict, "messages": self.messages}

    # def is_final_stage(self, step_idx):
    #     if self.params.final_stages is not None:
    #         val = self.params.pred_chains[step_idx] in self.params.final_stages
    #     else:
    #         # only the last step
    #         val = step_idx == len(stage_names) -1
    #     return val

    def omit_detailed_results(self):
        return "omit_detailed_results" in self.input_parameters_dict and self.input_parameters.omit_detailed_results == 1

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