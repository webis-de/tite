from argparse import ArgumentParser
from time import perf_counter

import ir_datasets
import pandas as pd
import torch
from torch.profiler import ProfilerActivity, profile, record_function
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer, BertConfig, PreTrainedModel, PreTrainedTokenizerBase

from tite.callbacks import DummyImportCallback
from tite.model import TiteConfig


def grad_docs(dataset: ir_datasets.Dataset, num_docs: int) -> list[str]:
    docs = []
    for doc in dataset.docs_iter():
        docs.append(doc.default_text())
        if len(docs) == num_docs:
            break
    return docs


def grab_queries(dataset: ir_datasets.Dataset, num_queries: int) -> list[str]:
    queries = []
    for query in dataset.queries_iter():
        queries.append(query.text)
        if len(queries) == num_queries:
            break
    return queries


def determine_batch_size(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, text: list[str], max_length: int
) -> int:
    batch_size = 2**12
    while True:
        try:
            encoding = tokenizer(
                text[:batch_size], max_length=max_length, padding=True, truncation=True, return_tensors="pt"
            )
            model(**encoding.to(model.device))
            break
        except RuntimeError as e:
            batch_size //= 2
            if batch_size == 1:
                raise e
    return batch_size // 2


def run_model(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, text: list[str], max_length: int, profile_func: bool
) -> pd.Series:
    batch_size = determine_batch_size(model, tokenizer, text, max_length)
    # ceil div
    num_batches = (len(text) + batch_size - 1) // batch_size

    def _run_model():
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        mem = torch.cuda.max_memory_allocated()
        elapsed = 0
        for i in tqdm(range(num_batches), position=3, leave=False):
            torch.cuda.synchronize()
            batch = text[i * batch_size : (i + 1) * batch_size]
            with record_function("tokenization"):
                encoding = tokenizer(
                    batch, max_length=max_length, padding=True, truncation=True, return_tensors="pt"
                ).to(model.device)
            torch.cuda.synchronize()
            start.record()
            with record_function("model inference"):
                model(**encoding.to(model.device))
            torch.cuda.synchronize()
            end.record()
            torch.cuda.synchronize()
            elapsed += start.elapsed_time(end) * 1e-3
        mem = torch.cuda.max_memory_allocated() - mem
        return elapsed, mem

    if profile_func:
        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
        sort_by_keyword = "cpu_time_total"
        with profile(activities=activities, record_shapes=True) as prof:
            elapsed, mem = _run_model()
        print(prof.key_averages().table(sort_by=sort_by_keyword, row_limit=10))
    else:
        elapsed, mem = _run_model()

    return pd.Series({"batch_size": batch_size, "num_batches": num_batches, "time": elapsed, "mem": mem})


def main(args=None):
    parser = ArgumentParser()
    parser.add_argument("--num_texts", type=int, default=10_000)
    parser.add_argument("--profile", action="store_true")

    args = parser.parse_args(args)

    dataset = ir_datasets.load("msmarco-passage/train")
    docs = grad_docs(dataset, args.num_texts)
    queries = grab_queries(dataset, args.num_texts)

    _results = {}

    # ("mini-lm-l6", AutoConfig.from_pretrained("sentence-transformers/msmarco-MiniLM-L-6-v3")),
    # ("mini-lm-l12", AutoConfig.from_pretrained("sentence-transformers/msmarco-MiniLM-L12-cos-v5")),
    configs = [
        ("bert", AutoConfig.from_pretrained("sentence-transformers/msmarco-bert-base-dot-v5")),
        # ("bert-large", AutoConfig.from_pretrained("bert-large-uncased")),
        ("modern-bert", AutoConfig.from_pretrained("answerdotai/ModernBERT-base")),
        ("distil-bert", AutoConfig.from_pretrained("sentence-transformers/msmarco-distilbert-dot-v5")),
        ("tite-bert-absolute", TiteConfig(positional_embedding_type="absolute")),
        ("tite-bert-rope", TiteConfig(positional_embedding_type="rotary")),
        (
            "tite-2-late-rope",
            TiteConfig(
                strides=(None, None, None, 2, 2, 2, 2, 2, 2, 2, 2, 2),
                kernel_sizes=(None, None, None, 2, 2, 2, 2, 2, 2, 2, 2, 2),
                positional_embedding_type="rotary",
            ),
        ),
        (
            "tite-2-late-rope-pre",
            TiteConfig(
                strides=(None, None, None, 2, 2, 2, 2, 2, 2, 2, 2, 2),
                kernel_sizes=(None, None, None, 2, 2, 2, 2, 2, 2, 2, 2, 2),
                positional_embedding_type="rotary",
                pooling_location="pre",
            ),
        ),
        (
            "tite-2-late-rope-post",
            TiteConfig(
                strides=(None, None, None, 2, 2, 2, 2, 2, 2, 2, 2, 2),
                kernel_sizes=(None, None, None, 2, 2, 2, 2, 2, 2, 2, 2, 2),
                positional_embedding_type="rotary",
                pooling_location="post",
            ),
        ),
        (
            "tite-3-late-rope",
            TiteConfig(
                strides=(None, None, None, None, None, None, 3, 3, 3, 3, 3, 3),
                kernel_sizes=(None, None, None, None, None, None, 3, 3, 3, 3, 3, 3),
                positional_embedding_type="rotary",
            ),
        ),
        (
            "tite-2-staggered-rope",
            TiteConfig(
                strides=(None, 2, 2, 2, None, 2, 2, 2, None, 2, 2, 2),
                kernel_sizes=(None, 2, 2, 2, None, 2, 2, 2, None, 2, 2, 2),
                positional_embedding_type="rotary",
            ),
        ),
        (
            "tite-3-staggered-rope",
            TiteConfig(
                strides=(None, 3, None, 3, None, 3, None, 3, None, 3, None, 3),
                kernel_sizes=(None, 3, None, 3, None, 3, None, 3, None, 3, None, 3),
                positional_embedding_type="rotary",
            ),
        ),
    ]

    pg = tqdm(configs)
    for config_name, config in pg:
        pg.set_description(config_name)
        model = AutoModel.from_config(config, torch_dtype=torch.bfloat16).eval().to("cuda")
        if model.config.name_or_path:
            tokenizer = AutoTokenizer.from_pretrained(model.config.name_or_path)
        else:
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        iterator, total = zip(["docs", "queries"], [docs, queries], [512, 32]), 2
        # iterator, total = zip(["queries"], [queries], [32]), 1
        # iterator, total = zip(["docs"], [docs], [512]), 1
        for text_type, text, max_length in tqdm(iterator, position=1, leave=False, total=total):
            for grad in tqdm((True, False), position=2, leave=False):
                # for grad in tqdm((False,), position=2, leave=False):
                # for grad in tqdm((True,), position=2, leave=False):
                if grad:
                    model_results = run_model(model, tokenizer, text, max_length, args.profile)
                else:
                    with torch.inference_mode():
                        model_results = run_model(model, tokenizer, text, max_length, args.profile)
                _results[(config_name, text_type, grad)] = model_results

    results = (
        pd.DataFrame(_results)
        .T.reset_index()
        .rename(columns={"level_0": "model", "level_1": "text_type", "level_2": "grad"})
    )
    results["num_texts"] = results["batch_size"] * results["num_batches"]
    results["num_texts_per_sec"] = results["num_texts"] / results["time"]
    results["kb / text"] = results["mem"] / results["num_texts"] / 1024
    results.to_json("efficiency.json")
    columns = ["text_type", "grad"]
    index = ["model"]
    values = ["num_texts_per_sec", "kb / text"]
    print(
        results.pivot_table(values=values, index=index, columns=columns)
        .reorder_levels([1, 2, 0], axis=1)
        .sort_index(axis=1)
        .round()
        .astype(int)
    )


if __name__ == "__main__":
    main()
