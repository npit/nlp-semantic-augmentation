## Installation
The process below installs code and resources in the "smaug" folder.

# Code
Get the sources -- use the `nviv` branch of the `npit/nlp-semantic-augmentation` repository, i.e.:
`git clone https://github.com/npit/nlp-semantic-augmentation/tree/nviv` ./smaug


# Dependencies
Install the package dependencies stored in `package-dependencies.txt`.
```sudo apt install $(cat package-dependencies.txt)```

Install the python3 package dependencies stored in `dependencies.txt`:
```pip3 install --user -r dependencies.txt```


## Execution resources and models

# Representations
Fetch the greek w2v embeddings.
```
mkdir -p smaug/raw_data/representation
cd smaug/raw_data/representation
# Fetch and extract
sudo apt install tar gzip
wget "http://archive.aueb.gr:8085/files/grcorpus_def.vec.gz" -O embeddings.gz && gunzip embeddings.gz > embeddings.csv
# Remove problematic entries (trailing whitespace)
 sed -i "s/ $//g" embeddings.csv
 sed -i "s/[\!(]//g" embeddings.csv
 sed -i "s/^[ ]//g" embeddings.csv
# Remove header and rename
tail -n +2 embeddings.csv > greek_word2vec.csv
rm embeddings.csv
```

# Classifiers
Fetch the pretrained classifiers:
```
wget https://users.iit.demokritos.gr/~pittarasnikif/nv/models.tar.gz && tar xzf models.tar.gz
```
Place the `.model` files and their wrappers in the same directory.


## Deployment

Use the `nv-ngram-twostage.yml.example` configuration file.
```
cp nv-ngram-twostage.yml.example nv-ngram-twostage.yml
```
# Representations
Enter the pretrained embeddings basename to the configuration field sequence:
- `chains` - `rep` - `representation` -`name`
E.g. for the embeddings file in `smaug/raw_data/representation/greek_w2v.csv` the above value should be `greek_w2v`

# Classifiers
Modify the configuration file with the correct paths to the pretrained classifier files, i.e. the entries
under the following two field sequencies:
- `chains` - `lrn_binary` - `model_path`
- `chains` - `lrn_multiclass` - `model_path`
Use just the path to the `.model` file, making suer the corresponding `.model.wrapper` lies in the same directory.

## Execution
Run the API via:
`python3 main.py nv-twostage.yml`
## Rest call examples
For API invokation examples and explanation, `rest-examples.sh`

## Codebase update
Just update the branch via git:
`git pull origin nviv`

- example tou call (curl calls in bash)
- odhgies gia code update (pull, repo)
- odhgies gia model update
- look into swagger (documentation gia REST apis)
- REST codes, inject swagger

- list package dependencies

12:00 monday charis
