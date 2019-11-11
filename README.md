# Text classification with semantic enrichment tool

## Dependencies

Install dependencies with:

`pip3 install -r dependencies`

If tensorflow keeps crashing on import, install version `1.5.0` with:

`pip3 install tensorflow==1.5.0`


## Run
Run a one-time experiment using the default example configuration `config.example.yml`:
```python
python3 main.py
```
Given that the example configuration uses a tiny subset of glove embeddings and very small portion of 20-Newsgroups dataset, awful performance is expected.
For a real-world classification run, please supply a configuration file:
```
python3 main.py myconfig.yml
```
Instructions on writing the configuration are presented below.

## Configuration
Configuration is performed via the `.yml` file. Common configuration parameters are outlined below:

### Dataset
Specify the dataset to use in the run. Publicly available distributions of [Reuters](https://martin-thoma.com/nlp-reuters/) and [20Newsgroups](http://qwone.com/~jason/20Newsgroups/) are supported, via python library API implementations. Additionally, custom datasets in json format are processable, using the layout specified in the `ManualDataset` class. Instance and class limiter parameters are available for testing purposes.

| parameter | candidate values | description |
| :--- | --- | --- |
| name  | `reuters`, `20newsgroups`, `path/to/mydataset.json` | Input dataset name / path specification.  |
| data_limit  | `[<int>, <int>]` | Number of instances to limit train and test data to. |
| class_limit  | `<int>` | Number of classes to limit the dataset to. |

### Embedding
Specifies the word embedding and text vector representation parameters to be utilized. Embeddings should be in the format of pickled pandas dataframes in the format specified below, and lie in the `raw_data/embeddings` folder. Unknown tokens keys are ignored unless specified and if they are missing from the input embedding map, they are initialized to zero vectors.

| parameter | candidate values | description |
| :--- | --- | --- |
|  name | `<embedding_name>_dim<dim>.pickle`| Embedding pickled dataframe, lying in `raw_data/embeddings/` folder. `<dim>` should match the vector dimension. |
|  aggregation |  `avg`, `pad` | How to combine word vectors into document vectors. |
|  sequence_length |  `<int>` | Length of word sequence to consider, for compatible aggregations. |
|  dimension |  `<dim>` | Embedding vector dimension. |
|  unknown_words |  `unk` | Key for the vector to assign missing words to. |

### Semantic
Specifies the semantic extraction and augmentation process, with [Wordnet](https://wordnet.princeton.edu/) being supported at this stage. Synset frequency and tf-idf weight vectors are supported, along with limiting the resulting set to the top `k` most prominent terms. Combination with the lexical embeddings can be done via concatenation or replacement.

| parameter | candidate values | description |
| :--- | --- | --- |
|  name | `wordnet` | The semantic resource name. |
|  unit | `concept` | The semantic information unit. `concept` is only supported for now. |
|  weights |  `tfidf`, `frequencies` | Type of semantic unit weights to build. |
|  limit | `[first | frequency, <int>]` | Vector pruning options: consider only the first (top) `k` concepts, or the ones appear at least `frequency` times in the dataset. |
| enrichment | `replace`, `concat` | Combination methods with the lexical component. |
| disambiguation | `pos, first, context_embedding` | Disambiguation procedure to map a single word to a semantic unit. |
| context\_file | `/path/to/contextfile.pickle` | File with concept context word vectors.  |
| context\_aggregation | `avg` | How to combine context word vectors. |
| context\_threshold | `<int>` | Frequency cutoff threshold.  |
| spreading\_activation | `[step, decay]` | Spreading activation step and decay.  |

For context-embedding disambiguation, the necessary files have to be produced first via the `extract_wordnet_synset_words.py` script.

###  Learner
Specifies the learning model to perform classification. MLP and LSTM neural networks are supported via [Keras](https://keras.io/), along with number of layers and neurons per layer configurations for the network. Note that the former requires single-vector instances and the latter a sequence of vectors per instance; therefore, LSTM should be used with `pad` embedding aggregation to generate multi-word tensors for its input.

| parameter | candidate values | description |
| :--- | --- | --- |
|  name | `mpl`, `lstm` | Classifier name. |
|  layers | `<int>` | Number of layers. |
|  hidden_dim | `<int>` | Number of neurons per layer. |

###  Train
Determines the learner training and validation parameters.

| parameter | candidate values | description |
| :--- | --- | --- |
| epochs |  `<int>` | Maximum number of epochs to train for. |
| folds |  `<int>` | Number of folds to split the training set to. |
| validation_portion |  `<float>` | Portion of the training set to use as validation. Should lie in `(0, 1)`. |
| early_stopping_patience |  `<int>` | Number of epochs to wait before early stopping at validation loss stagnation. |
| batch_size |  `<int>` | Number of instances per training / testing batch. |


### Print
Controls measure, label aggregations and baseline run types to pring at the end of / during training.


| parameter | candidate values | description |
| :--- | --- | --- |
| measures |  `f1-score, precision, recall, accuracy` | Classification evaluation measures. |
| aggregations |  `macro, micro, weighted` | Label multiclass aggregation. Incompatible measure-aggregation combinations are ignored. |
| run_types |  `run, majority, random` | The actual run, along with majority (with respect to the most populous class in the training set) and random baseline runs. |
| stats |  `mean, var, std` | Statistics to summarize performance accross folds. |

### Folders
Controls folder configuration for the run execution.

| parameter | candidate values | description |
| :--- | --- | --- |
|  run | `/path/to/run/folder/` | Where to store run logs and results. | 
|  serialization |  `/path/to/serialization/folder/` | Where to look for and store intermmediate results. Default: `serialization` |
|  raw_data | `/path/to/raw_data/folder/` | Where to look for raw data. Default: `raw_data` |

<!-- Run large-scale experiments with [this wrapper](https://github.com/npit/nlesi-neural-augmentation). -->
