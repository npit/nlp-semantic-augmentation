# Installation
The process below installs code and resources in the "smaug" folder.

## Code
You need to have git

```
sudo apt update && sudo apt install -y git
```

Get the sources -- use the `nviv` branch of the `npit/nlp-semantic-augmentation` repository, i.e.:

```
git clone https://github.com/npit/nlp-semantic-augmentation ./smaug 
cd smaug && git checkout nviv
```


## Dependencies
### System Packages
Install the package dependencies stored in `package-dependencies.txt`.

```
sudo apt update && sudo apt install -y $(cat package-dependencies.txt)
```

### Python 3.8
To install the latest python environment for older ubuntu-based systems:
Install the package:

```
sudo apt install -y software-properties-common curl

sudo add-apt-repository ppa:deadsnakes/ppa && sudo apt update
sudo apt install -y python3.8 python3.8-distutils

```
Get the relevant pip vers.:
```
sudo apt remove python3-pip
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3.8 get-pip.py
python3.8 -m pip install --upgrade pip

```
*Note*: Use `python3.8` in all following commands (and/or use aliasing).

### Python Packages
Install the python package dependencies stored in `dependencies.txt`:

```
python3 -m pip install --user -r dependencies.txt
```


## Execution resources and models

### Representations
Fetch the greek w2v embeddings.

```
mkdir -p raw_data/representation && cd $_
# Fetch and extract
sudo apt -y install tar gzip wget
wget "http://archive.aueb.gr:8085/files/grcorpus_def.vec.gz" -O embeddings.gz && gunzip -c embeddings.gz > embeddings.csv && rm embeddings.gz
# Remove problematic entries (trailing whitespace and problematic tokens)
 sed -i "s/ $//g" embeddings.csv
 sed -i "s/[\!()\#]//g" embeddings.csv
 sed -i "/^ /d" embeddings.csv
# Remove header and rename
tail -n +2 embeddings.csv > greek_word2vec.csv
rm embeddings.csv
```

### Classifiers
Fetch the pretrained classifiers:
```
cd ../..
wget https://users.iit.demokritos.gr/~pittarasnikif/nv/models.tar.gz && tar xzf models.tar.gz
wget  -O labelmapping.json https://users.iit.demokritos.gr/~pittarasnikif/nv/multiclass_labelnames.json
```


# Deployment
## Configuration
Use the preconfigured configuration file:
```
cp nv-ngram-twostage.yml.example nv-ngram-twostage.yml
```
## Execution
Run the API via:

```
python3 main.py nv-twostage.yml
```

## Rest call examples
For API invokation examples and explanation, see `rest-examples.sh`

# Maintenance
## Codebase update
Just update the branch via git:
```
git pull origin nviv
```

## Changing pretrained models

### Representations
The pretrained embeddings basename should be set to the configuration field sequence:
- `chains` - `rep` - `representation` -`name`
   E.g. for the embeddings file in `smaug/raw_data/representation/greek_w2v.csv` the above value should be `greek_w2v`
- Make sure problematic entries are handled, as detailed above

### Classifiers
The pretrained classifier files should be set in the configuration file under the following two field sequencies:
- `chains` - `lrn_binary` - `model_path`
- `chains` - `lrn_multiclass` - `model_path`
Place the `.model` files and their `.wrapper` counterparts in the same directory.

# Info
## Data, training and model details
The data used to build the pretrained classifiers were generated using [the tools in this repository](https://gitlab.com/npit/word-level-transcription-classification), using parameters are as below:
- Preprocessing:
   - Data were split to ngrams, using a before/after context (around each word) of 3 tokens max, discarding data with no available context (i.e. single-word instances).
   - Lowercasing & punctuation removal
   - Sequence alignment: Truncation of longer sequence in trasncription / gt pairs
- Word embeddings: Greek word2vec (http://archive.aueb.gr:7000/resources/). Concatenation of before-context, center and after-context portions of the data.
- Classifiers: Logistic regresion for both binary and multiclass classification.

# TODOs
See `rest-examples.sh`. Will provide swagger-based documentation in the future.
