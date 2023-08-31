import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from utility import get_logger, log_model_config

import os

logger = get_logger()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, required=True, help="location of json data"
    )
    parser.add_argument(
        "--model_dir", type=str, required=True, help="where to store vector db"
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    args.model_dir = os.path.join(args.model_dir, timestamp)

    return args


if __name__ == "__main__":
    args = parse_args()

    logger.info(f"Loading data from {args.dataset}")
    data: List[Dict] = json.loads(Path(args.dataset).read_text())
    docs = [
        Document(page_content=d["page_content"], metadata=d["metadata"]) for d in data
    ]

    logger.info(f"Generating vector db to be stored at {args.model_dir}")
    db = DeepLake.from_documents(
        docs,
        OpenAIEmbeddings(disallowed_special=()),
        dataset_path=args.model_dir,
        overwrite=True,
    )
    log_model_config(
        logger=logger,
        args=args,
        log_dir="/workspaces/projects/code_understanding/model",
    )
