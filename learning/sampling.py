"""Perform sampling modifications"""
from imblearn.over_sampling import RandomOverSampler

from collections import Counter

def oversample_single_sample_labels(data, labels, target_num=2):
    """
    Return a dict where keys are single-sample labels and values the desired number to reach
    """
    cn = Counter(labels)
    singles = [x[0] for x in cn.most_common() if x[1] == 1]
    if not singles:
        return data, labels
    desired_dict = {k: target_num for k in singles}
    return oversample(data, labels, desired_dict)

def oversample(data, labels, label_desired_nums_dict):
    """
    Perform oversampling by specified label sample counts

    label_desired_nums_dict (dict): Dictionary of entries label: desired num
    """
    import ipdb; ipdb.set_trace()
    sampler = RandomOverSampler(sampling_strategy=label_desired_nums_dict)
    rx, ry = sampler.fit_resample(data.reshape(-1, 1), labels)
    rx = rx.squeeze()
    return rx, ry

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
