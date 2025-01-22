# TITE

This repository contains the code for the paper "TITE: Token-Independent Text Encoder for Information Retrieval"

The `tite` directory contains all code for modeling, data processing, and pre-training. The `main.py` script is the entrypoint for pre-training and uses [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) to manage training.

Fine-tuning is done using [Lightning IR]([https://](https://webis.de/lightning-ir/index.html)). The configuration files for fine-tuning are in the `lightning-ir-configs` directory.
