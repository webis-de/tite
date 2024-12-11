from argparse import ArgumentParser
from pathlib import Path

import ir_datasets
from pyserini.search.lucene import LuceneSearcher
from tqdm import tqdm


def main(args=None):

    parser = ArgumentParser()

    parser.add_argument("--save_dir", type=Path, default=Path.cwd() / "runs" / "bm25")
    parser.add_argument("--dev_datasets", type=str, nargs="+", default=None)

    args = parser.parse_args(args)

    dev_datasets = args.dev_datasets

    if dev_datasets is None:
        dev_datasets = [
            "beir/dbpedia-entity/dev",
            "beir/fever/dev",
            "beir/fiqa/dev",
            "beir/hotpotqa/dev",
            "beir/nfcorpus/dev",
            "beir/quora/dev",
        ]

    for dataset in tqdm(dev_datasets):
        name = dataset.split("/")[1]
        searcher = LuceneSearcher.from_prebuilt_index(f"beir-v1.0.0-{name}.flat")
        dataset = ir_datasets.load(dataset)
        save_path = args.save_dir / f"{dataset.dataset_id().replace('/', '-')}.run"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("w") as f:
            for query in tqdm(dataset.queries_iter(), total=dataset.queries_count(), position=1):
                hits = searcher.search(query.text, k=100)
                for i, hit in enumerate(hits):
                    f.write(f"{query.query_id} Q0 {hit.docid} {i+1} {hit.score} bm25\n")


if __name__ == "__main__":
    main()
