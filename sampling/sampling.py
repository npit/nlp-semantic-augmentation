"""Perform sampling modifications"""
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
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

        self.max_label_val = 999
        self.exclusion_func = lambda x: x - (self.max_label_val + 1)
        self.restoration_func = lambda x: x + (self.max_label_val + 1)

        self.excluded_labelset = set()

    def remove_tag_exclusion(self, labels):
        new_excluded_idx = [i for (i,x) in enumerate(labels) if x < 0]
        ii = np.concatenate([np.where(self.reference_index ==x)[0] for x in self.exclude_idx])
        error("Tag exclusion index error.", sorted(ii) != new_excluded_idx)
        for x in ii:
            labels[x] = self.restoration_func(labels[x])

        error("Tag exclusion value error.", not np.all(labels[ii] == self.original_excluded_values))

    def apply_tag_exclusion(self, labels):
        """Mark labels for exclusion, wrt. specified tags to exclude""" 
        error("Negative labels exist, which prevents label exclusion", np.any(labels < 0))
        self.max_label_val = labels.max()
        self.exclude_idx = np.asarray([], dtype=np.int32)
        self.original_excluded_values = np.asarray([], dtype=np.int32)
        for tag in self.config.exclude_tags:
            idx = self.labels_dp.get_usage(Indices).get_tag_instances(tag)
            info(f"Excluding tag [{tag}], with {len(idx)} samples and distro {self.get_label_distro(labels[idx])}")
            self.exclude_idx = np.append(self.exclude_idx, idx)
            self.original_excluded_values = np.append(self.original_excluded_values, labels[idx])
        # set dummy values
        labels[self.exclude_idx] = self.exclusion_func(labels[self.exclude_idx])
        self.excluded_values = labels[self.exclude_idx]
        self.excluded_labelset.update(self.excluded_values)
        return labels

    def get_component_inputs(self):
        # self.inputs = self.data_pool.request_data(None, GroundTruth, self.name, "subset")
        # fetch all inputs
        inputs = self.data_pool.get_current_inputs()
        # must have only one dpack with a ground truth usage
        # on which label-wise sampling will be performed
        self.labels_dp = self.data_pool.request_data(None, Labels, self.name, "subset") 
        # only single-label
        error(f"Cannot apply {self.name} sampling to multilabel annotations.", self.labels_dp.get_usage(Labels).multilabel)

        # convert to a single ndarray
        labels = np.concatenate(self.labels_dp.data.instances)
        if self.config.exclude_tags is not None:
            labels = self.apply_tag_exclusion(labels)
        else:
            labels = self.labels_dp.data.instances

        self.original_labels = self.reference_labels = labels
        self.original_index = self.reference_index = np.arange(len(self.reference_labels))
        # resulting index-based transformation will be applied to all other input dpacks
        self.data = self.data_pool.request_data(None, None, self.name, usage_matching="any", usage_exclude=Labels, must_be_single=False) 
        # input length has to match
        lengths = [len(x.data.instances) for x in (self.data + [self.labels_dp])]
        if not len(set(lengths)) == 1:
            error(f"Inputs to {self.name} sampling with different lengths: {lengths}.")


    def get_label_distro(self, labels):
        cn = Counter(labels)
        return [x for x in cn.most_common() if x[0] not in self.excluded_labelset]


    def produce_outputs(self):
        self.output_labels = self.labels_dp.get_copy()
        labels_usage = self.labels_dp.get_usage(Labels)
        labelset = range(len(labels_usage.label_names))
        self.outputs = []
        if any(x is not None for x in (self.config.min_freq, self.config.max_freq)):
            sampling_dicts, distro = self.make_label_transformation_dict(self.reference_labels, self.config.min_freq, self.config.max_freq)
            info(f"Label size / distribution prior to resampling: {len(self.reference_labels)},  {distro}")

            if not any(sampling_dicts.values()):
                # no sampling required
                info(f"No under/oversampling required to reach max/min label freqs of {self.config.max_freq, self.config.min_freq}!")
                self.outputs = self.data
                return

            affected_labels = [p for k in sampling_dicts.values() for p in k]
            unaffected_labels = [l for l in labelset if l not in [p for k in sampling_dicts.values() for p in k]]
            msg = f"Will modify labels: {affected_labels}" + ("" if unaffected_labels else f"No need to resample labels: {unaffected_labels}")

            # apply resampling -- first undersampling, then oversampling
            for oper in "undersample oversample".split():
                ddict = sampling_dicts[oper]
                info(f"Applying {oper} on {len(ddict)} label(s): {list(ddict.keys())}")
                for l in ddict:
                    debug(f"Label: {l} {oper}: {dict(distro)[l]} -> {ddict[l]}")

                self.reference_index, self.reference_labels = self.resample(oper, self.reference_index, self.reference_labels, ddict)
                distro = self.get_label_distro(self.reference_labels)
                info(f"New distribution: {distro} / shape {self.reference_index.shape}")

            self.output_labels.apply_index_change(self.reference_index)
            if self.config.exclude_tags is not None:
                self.remove_tag_exclusion(self.reference_labels)


            info(f"Done resampling -- applying changes to the rest of {len(self.data)} input datapacks")
            # apply the resampling
            for inp in self.data:
                debug(f"{inp}")
                # udpate new data configuration
                dp_copy = inp.get_copy()
                dp_copy.apply_index_change(self.reference_index)
                self.outputs.append(dp_copy)

            ls = self.output_labels.data.instances
            lns = labels_usage.map_to_label_names(np.concatenate(ls))
            dats = [x['words'][0] for x in self.data[0].data.instances]
            for d, l in zip(dats,lns):
                print(d, l)
        else:
            error(f"Undefined desired operation for {self.name}")

    def set_component_outputs(self):
        self.data_pool.add_data_packs(self.outputs + [self.output_labels], self.name)


    def make_label_transformation_dict(self, labels, min_freq=None, max_freq=None):
        """
        Return a dict where keys are labels and values the desired number to reach
        """
        distro = self.get_label_distro(labels)
        unders_candidates, overs_candidates = [], []
        if min_freq:
            overs_candidates = [x[0] for x in distro if x[1] < min_freq]
        if max_freq:
            unders_candidates = [x[0] for x in distro if x[1] > max_freq]
            both = [x for x in overs_candidates if x in unders_candidates]
            error(f"Labels {both} specified for both over/undersampling", both)
            if not overs_candidates:
                info(f"Min freq: {self.config.min_freq} is already large enough: {distro[-1]}")
            if not unders_candidates:
                info(f"Max freq: {self.config.max_freq} is already low enough -- least populous class count: {distro[0]}")

        overs_dict = {k: min_freq for k in overs_candidates}
        unders_dict = {k: max_freq for k in unders_candidates}
        return {"oversample": overs_dict, "undersample": unders_dict}, distro

    def resample(self, op, data, labels, label_desired_nums_dict):
        """Perform under/oversampling to reach the specified label sample counts

           op (str): Operation identifier "oversample" or "undersample"
           label_desired_nums_dict (dict): Dictionary of entries label: desired num
           data (iterable): Data container
           data (iterable): Labels container
        """
        if type(data) != np.ndarray:
            data = np.asarray(data)
        if op == "oversample":
            sampler = RandomOverSampler(sampling_strategy=label_desired_nums_dict)
        elif op == "undersample":
            sampler = RandomUnderSampler(sampling_strategy=label_desired_nums_dict)
        else:
            error(f"Undefined resampling operation {op}")
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
