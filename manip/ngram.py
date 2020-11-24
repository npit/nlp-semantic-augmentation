# from manip.filter import Filter
from manip.manip import Manipulation
from utils import info, error, is_collection, make_indexes
from collections import OrderedDict
from bundle.datatypes import *
from bundle.datausages import *
from dataset.dataset import Dataset

class NGram(Manipulation):
    """Class to expand inputs to ngram instances
    Creates center, before-context and after-context
    """
    name = "ngram"
    # surrounding window size
    before, after = None, None

    def __init__(self, config):
        self.config = config
        Manipulation.__init__(self)
        # Filter.__init__(self, config)
        win_sizes = self.config.size if self.config.size is not None else [3, 3]
        if not is_collection(win_sizes):
            win_sizes = [win_sizes] * 2
        self.before, self.after = win_sizes

    def apply_operation(self, inputs):
        """Generate ngrams from input sequences
        """
        # gather contexts separately to make subsequent flexible handling doable
        before, after, center = [], [], []
        out_train, out_test = [], []
        train, test = self.indices.get_train_test()
        global_idx = -1
        # keep track to which instance each center word belongs to
        instance_level_index = []
        # produce ngrams
        for s, sequence in enumerate(inputs):
            sequence = Dataset.get_words(sequence)
            instance_level_index.extend([s] * len(sequence))
            for i in range(len(sequence)):
                global_idx += 1
                # get and filter indices for fefore - after contexts
                cbi = [x for x in range(i-self.before, i) if x >= 0]
                cai = [x for x in range(i+1, i+self.after) if x < len(sequence)]
                # assign elements
                cb = [sequence[k] for k in cbi]
                ca = [sequence[k] for k in cai]
                # add to collections
                before.append(cb)
                center.append(sequence[i])
                after.append(ca)
                # handle role indexing to the new collection
                if s in train:
                    out_train.append(global_idx)
                if s in test:
                    out_test.append(global_idx)
        # all data in a single container
        data = [Dataset.get_instance_from_words(x) for x in (center + before + after)]
        # all indexes
        all_tags = [defs.roles.train, defs.roles.test] + "center before after".split()
        # add train-test idxs, expanding to center, before and after sections
        expand_idx = lambda idx: idx + [x + k* len(center) for k in (1, 2) for x in idx]
        all_idxs = [expand_idx(out_train), expand_idx(out_test)] 
        all_idxs.extend(make_indexes([len(x) for x in (center, before, after)]))

        # add the word2instance indexes:
        inst_idx_dict = {}
        # duplicate for context
        instance_level_index *= 3
        for inst in set(instance_level_index):
            idx = [i for (i, idx) in enumerate(instance_level_index) if idx == inst]
            all_tags.append(f"ngram_inst_{inst}")
            all_idxs.append(idx)
        # all_idxs.append(instance_level_index)
        # all_tags.append("ngram")

        self.indexes = Indices(all_idxs, tags=all_tags)
        self.output = Text(data)

    def get_component_inputs(self):
        super().get_component_inputs()
        error(f"{self.name} can only operate on a single input, but {len(self.inputs)}", len(self.inputs) > 1)
        self.inputs = self.inputs[0]
        self.indices = self.indices[0]


    def set_component_outputs(self):
        dp = DataPack(self.output, usage=self.indexes)
        self.data_pool.add_data_packs([dp], self.name)


    def produce_outputs(self):
        info(f"Generating ngrams with window sizes {(self.before, self.after)}")
        self.apply_operation(self.inputs)