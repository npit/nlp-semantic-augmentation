# Text classification with semantic enrichment tool

## Dependencies

Install dependencies with:

`pip3 install $(cat dependencies)`

If tensorflow keeps crashing on import, install version `1.5.0` with:

`pip3 install tensorflow==1.5.0`


## Run
Run a one-time experiment:
```python
# run with default config file config.yml
python3 main.py

# else, specify
python3 main.py myconfig.yml
```

Run large-scale experiments with [this wrapper](https://github.com/npit/nlesi-neural-augmentation).