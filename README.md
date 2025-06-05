# TITE

This repository contains the code for the paper [`TITE: Token-Independent Text Encoder for Information Retrieval`](https://downloads.webis.de/publications/papers/schlatt_2025d.pdf).

## Model Zoo

Coming soon

## Pre-training

To pre-train a TITE model, run the following command:

```bash
python main.py \
  --config configs/trainer.yaml \
  --config configs/adamw.yaml \
  --config configs/data/datamodule-tite.yaml \
  --config configs/model/tite.yaml
```

See the configuration files in the `configs` directory for available configuration options. 

Note that this command will stream the Hugging Face FineWeb dataset. We recommend first downloading the dataset and then using the local path as exemplified below to avoid streaming the dataset from Hugging Face during training:

```yaml
data:
  class_path: tite.datasets.FineWebDataModule
  init_args:
    path: arrow
    data_files:
      train: ./HuggingFaceFW___fineweb-edu/default/0.0.0/*/fineweb-edu-train-*.arrow
```

## Fine-tuning

We rely on the [Lightning IR](https://lightning-ir.webis.de/) framework for fine-tuning, indexing, and retrieval. To fine-tune a TITE model, first insert the path or id of a pre-trained model in the `lightning-ir-configs/fine-tune/model.yaml` file. You can then use the following command:

```bash
lightning-ir fit \
  --config lightning-ir-configs/fine-tune/trainer.yaml \
  --config lightning-ir-configs/fine-tune/model.yaml \
  --config lightning-ir-configs/fine-tune/datamodule.yaml \
  --config lightning-ir-configs/fine-tune/adamw.yaml
```

To evaluate a fine-tuned model, first insert the fine-tuned model name in the `lightning-ir-configs/index/model.yaml` and `lightning-ir-configs/search/model.yaml` files. You can then use the following commands to index and search:

```bash
lightning-ir index \
  --config lightning-ir-configs/index/trainer.yaml \
  --config lightning-ir-configs/index/model.yaml \
  --config lightning-ir-configs/index/datamodule.yaml \
  --config lightning-ir-configs/index/index-callback.yaml

lightning-ir search \
  --config lightning-ir-configs/search/trainer.yaml \
  --config lightning-ir-configs/search/model.yaml \
  --config lightning-ir-configs/search/datamodule.yaml \
  --config lightning-ir-configs/search/search-callback.yaml
```

## Reproduction

The run files to reproduce the tables in the paper are available on [Zenodo](https://zenodo.org/records/15603441). Download and unpack the run files and then run the `notebooks/evaluate.ipynb` notebook to reproduce the results.

The `efficiency.py` script can be used to reproduce the efficiency results. It outputs an `efficiency.json` file that can be copied into the `notebooks` directory and be evaluated by running the `notebooks/efficiency.ipynb` notebook.
