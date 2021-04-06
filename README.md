# PRISM rule inducer algorithm

## How to run

See options:
```bash
python3 main.py --help
```

Run PRISM on a dataset:
```bash
python3 main.py -d DATASET_NAME
```
This command splits the dataset into 2 (train and test sets), then induces the rules with PRISM on the train set, and finally evaluates the accuracy on the test set.
