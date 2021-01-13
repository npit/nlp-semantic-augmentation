"""Perform sampling modifications"""
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
from component.component import Component
from utils import error, info
from bundle.datausages import *
import numpy as np

class Instantiator:
    component_name = "sample"
    @staticmethod
    def create(config):
        return Sampler(config)

class Sampler(Component):
    name = "sampler"
    def __init__(self, config):
        self.config = config

    def get_component_inputs(self):

        # self.inputs = self.data_pool.request_data(None, GroundTruth, self.name, "subset")
        # fetch all inputs
        inputs = self.data_pool.get_current_inputs()
        # must have only one dpack with a ground truth usage
        # on which label-wise sampling will be performed
        self.labels_dp = self.data_pool.request_data(None, Labels, self.name, "subset") 
        # only single-label
        error(f"Cannot apply {self.name} sampling to multilabel annotations.", self.labels_dp.get_usage(Labels).multilabel)
        self.reference_labels = np.concatenate(self.labels_dp.data.instances)
        # resulting transformation will be applied to all other input dpacks
        self.data = self.data_pool.request_data(None, None, self.name, usage_matching="any", usage_exclude=Labels, must_be_single=False) 
        # input length has to match
        lengths = [len(x.data.instances) for x in (self.data + [self.labels_dp])]
        if not len(set(lengths)) == 1:
            error(f"Inputs to {self.name} sampling with different lengths: {lengths}.")


    def produce_outputs(self):
        self.output_labels = None
        self.outputs = []
        if self.config.min_freq is not None:
            new_labels = None
            ddict = self.make_label_transformation_dict(self.reference_labels, self.config.min_freq)
            if not ddict:
                # no oversampling required
                info(f"No oversampling required to reach a min label freq. of {self.config.min_freq}!")
                self.outputs = self.data + [self.labels_dp]
                return
            info(f"Will oversample {len(ddict)} / {len(self.labels_dp.get_usage(Labels).labelset)} labels, e.g. 10 of these: {list(ddict.keys())[:10]}")
            for inp in self.data:
                data_idx = np.arange(len(inp.data.instances))
                if new_labels is None:
                    new_data_idx, new_labels = self.oversample(data_idx, self.reference_labels, ddict)
                    new_labels = np.split(new_labels, len(new_labels))
                    added_idx = new_data_idx[len(data_idx):]

                    self.output_labels = DataPack(Numeric(new_labels), usage=self.labels_dp.usages)
                    self.output_labels.apply_index_expansion(added_idx, old_data_size=len(data_idx))

                info(f"Oversampling dpack {inp} {len(data_idx)} -> {len(new_data_idx)}")
                # make data dpack
                dat = DataPack(type(inp.data)(inp.data.get_slice(new_data_idx)), usage=inp.usages)
                dat.apply_index_expansion(added_idx, old_data_size=len(data_idx))
                self.outputs.append(dat)
        else:
            error(f"Undefined desired operation for {self.name}")

    def set_component_outputs(self):
        self.data_pool.add_data_packs(self.outputs + [self.output_labels], self.name)


    def make_label_transformation_dict(self, labels, min_freq=None):
        """
        Return a dict where keys are labels and values the desired number to reach
        """
        cn = Counter(labels)
        if min_freq:
            candidates = [x[0] for x in cn.most_common() if x[1] < min_freq]
            desired_dict = {k: min_freq for k in candidates}
        return desired_dict

    def oversample(self, data, labels, label_desired_nums_dict):
        """
        Perform oversampling by specified label sample counts

        label_desired_nums_dict (dict): Dictionary of entries label: desired num
        """
        if type(data) != np.ndarray:
            data = np.asarray(data)
        sampler = RandomOverSampler(sampling_strategy=label_desired_nums_dict)
        rx, ry = sampler.fit_resample(data.reshape(-1, 1), labels)
        rx = rx.squeeze()
        return rx, ry

class SSampler:
    def __init__(self):
        pass

    def sample(train_data, train_labels):
        # do sampling processing
        if self.do_sampling:
            for split_index, (tr, vl) in enumerate(splits):
                aug_indexes = []
                orig_tr_size = len(tr)
                ldaug = LabelledDataAugmentation()
                if self.sampling_method == defs.sampling.oversample:
                    for (label1, label2, ratio) in self.sampling_ratios:
                        aug_indexes.append(
                            ldaug.oversample_to_ratio(self.train,
                                                      self.train_labels,
                                                      [label1, label2],
                                                      ratio,
                                                      only_indexes=True,
                                                      limit_to_indexes=tr))
                        info(
                            "Sampled via {}, to ratio {}, for labels {},{}. Modification size: {} instances."
                            .format(self.sampling_method, ratio, label1,
                                    label2, len(aug_indexes[-1])))
                    aug_indexes = np.concatenate(aug_indexes)
                    tr = np.append(tr, aug_indexes)
                    info("Total size change: from {} to {} training instances".
                         format(orig_tr_size, len(tr)))
                elif self.sampling_method == defs.sampling.undersample:
                    for (label1, label2, ratio) in self.sampling_ratios:
                        aug_indexes.append(
                            ldaug.undersample_to_ratio(self.train_data,
                                                       self.train_labels,
                                                       [label1, label2],
                                                       ratio,
                                                       only_indexes=True))
                        info(
                            "Sampled via {}, to ratio {}, for labels {},{}. Modification size: {} instances."
                            .format(self.sampling_method, ratio, label1,
                                    label2, len(aug_indexes[-1])))
                    aug_indexes = np.concatenate(aug_indexes)
                    tr = np.delete(tr, aug_indexes)
                    info("Total size change: from {} to {} training instances".
                         format(orig_tr_size, len(tr)))
                else:
                    error(
                        "Undefined augmentation method: {} -- available are {}"
                        .format(self.sampling_method, defs.avail_sampling))
                splits[split_index] = (tr, vl)
            return splits
