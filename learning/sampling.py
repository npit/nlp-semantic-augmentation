"""Perform sampling modifications"""

class Sampler:
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
