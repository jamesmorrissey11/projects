import argparse
import json
import os
from datetime import datetime

from langchain.text_splitter import RecursiveCharacterTextSplitter
from utility import files_to_documents, get_logger


def clone_repository(repo_dir, repo_url):
    try:
        os.system(f"git clone {repo_url} {repo_dir}")
    except Exception as e:
        logger.info("Unable to clone repo")


def write_documents_to_json(docs, data_dir, json_path):
    split_json = [
        {"page_content": d.page_content, "metadata": d.metadata} for d in docs
    ]
    logger.info(f"Writing {len(split_json)} documents to {data_dir}")
    with open(json_path, "w") as f:
        json.dump(split_json, f)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo_dir", type=str, required=True, help="location to clone repo to"
    )
    parser.add_argument(
        "--data_dir", type=str, required=True, help="location to store documents at"
    )
    parser.add_argument(
        "--repo_url", type=str, required=True, help="url of github repo to clone"
    )
    parser.add_argument(
        "--download", type=str, default="no", help="whether or not to download the repo"
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    args.data_dir = os.path.join(args.data_dir, timestamp)
    os.mkdir(args.data_dir)

    args.json_path = os.path.join(args.data_dir, "data.json")

    return args


logger = get_logger()

if __name__ == "__main__":
    args = parse_args()
    if args.download:
        clone_repository(repo_dir=args.repo_dir, repo_url=args.repo_url)
    documents = files_to_documents(
        folder_name=args.repo_dir + "/libs/langchain/langchain"
    )
    logger.info(f"Splitting {len(documents)} documents")
    python_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    split_documents = python_splitter.split_documents(documents)
    write_documents_to_json(
        docs=split_documents, data_dir=args.data_dir, json_path=args.json_path
    )
