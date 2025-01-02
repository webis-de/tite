from argparse import ArgumentParser
from time import perf_counter

import ir_datasets
import pandas as pd
import torch
from torch.profiler import ProfilerActivity, profile
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
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    mem = torch.cuda.max_memory_allocated()

    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    sort_by_keyword = "cuda_time_total"

    with profile(activities=activities, record_shapes=True) as prof:
        start = perf_counter()
        for i in tqdm(range(num_batches), position=2, leave=False):
            batch = text[i * batch_size : (i + 1) * batch_size]
            encoding = tokenizer(batch, max_length=max_length, padding=True, truncation=True, return_tensors="pt")
            model(**encoding.to(model.device))
        torch.cuda.synchronize()
        elapsed = perf_counter() - start
        mem = torch.cuda.max_memory_allocated() - mem

    if profile_func:
        print(prof.key_averages(group_by_input_shape=True).table(sort_by=sort_by_keyword, row_limit=20))

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

    configs = [
        ("bert", AutoConfig.from_pretrained("sentence-transformers/msmarco-bert-base-dot-v5")),
        ("modern-bert", AutoConfig.from_pretrained("answerdotai/ModernBERT-base")),
        # ("mini-lm-l6", AutoConfig.from_pretrained("sentence-transformers/msmarco-MiniLM-L-6-v3")),
        # ("mini-lm-l12", AutoConfig.from_pretrained("sentence-transformers/msmarco-MiniLM-L12-cos-v5")),
        ("distil-bert", AutoConfig.from_pretrained("sentence-transformers/msmarco-distilbert-dot-v5")),
        ("tite-bert-absolute", TiteConfig(positional_embedding_type="absolute")),
        # ("tite-bert-alibi", TiteConfig(positional_embedding_type="ALiBi")),
        ("tite-bert-rope", TiteConfig(positional_embedding_type="rotary")),
        # (
        #     "tite-2-late-absolute",
        #     TiteConfig(
        #         stride=[None, None, None, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        #         kernel_size=[None, None, None, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        #         positional_embedding_type="absolute",
        #     ),
        # ),
        # (
        #     "tite-2-late-alibi",
        #     TiteConfig(
        #         stride=[None, None, None, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        #         kernel_size=[None, None, None, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        #         positional_embedding_type="ALiBi",
        #     ),
        # ),
        (
            "tite-2-late-rope",
            TiteConfig(
                stride=[None, None, None, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                kernel_size=[None, None, None, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                positional_embedding_type="rotary",
            ),
        ),
        (
            "tite-2-late-rope-pre",
            TiteConfig(
                stride=[None, None, None, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                kernel_size=[None, None, None, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                positional_embedding_type="rotary",
                pooling_location="pre",
            ),
        ),
        (
            "tite-2-late-rope-post",
            TiteConfig(
                stride=[None, None, None, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                kernel_size=[None, None, None, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                positional_embedding_type="rotary",
                pooling_location="post",
            ),
        ),
        (
            "tite-3-late-rope",
            TiteConfig(
                stride=[None, None, None, None, None, None, 3, 3, 3, 3, 3, 3],
                kernel_size=[None, None, None, None, None, None, 3, 3, 3, 3, 3, 3],
                positional_embedding_type="rotary",
            ),
        ),
        (
            "tite-2-staggered-rope",
            TiteConfig(
                stride=[None, 2, 2, 2, None, 2, 2, 2, None, 2, 2, 2],
                kernel_size=[None, 2, 2, 2, None, 2, 2, 2, None, 2, 2, 2],
                positional_embedding_type="rotary",
            ),
        ),
        (
            "tite-3-staggered-rope",
            TiteConfig(
                stride=[None, 3, None, 3, None, 3, None, 3, None, 3, None, 3],
                kernel_size=[None, 3, None, 3, None, 3, None, 3, None, 3, None, 3],
                positional_embedding_type="rotary",
            ),
        ),
    ]

    pg = tqdm(configs)
    for config_name, config in pg:
        pg.set_description(config_name)
        model = AutoModel.from_config(config).eval().to("cuda")
        if model.config.name_or_path:
            tokenizer = AutoTokenizer.from_pretrained(model.config.name_or_path)
        else:
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        for text_type, text, max_length in tqdm(
            zip(["docs", "queries"], [docs, queries], [512, 32]), position=1, leave=False, total=2
        ):
            for grad in tqdm((True, False), position=2, leave=False):
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
    columns = ["text_type", "grad"]
    index = ["model"]
    values = ["num_texts_per_sec", "kb / text"]
    print(
        results.pivot_table(values=values, index=index, columns=columns)
        .reorder_levels([1, 0], axis=1)
        .sort_index(axis=1)
        .round()
        .astype(int)
    )
    results.to_json("efficiency.json")


if __name__ == "__main__":
    main()
