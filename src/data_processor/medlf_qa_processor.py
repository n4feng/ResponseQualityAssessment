import os
import json

from src.rag.retrieval import DocDB
from src.data_processor.raw_data_processor import DatasetProcessor


class MedLFQAProcessor(DatasetProcessor):

    def process_queries(self, input_file: str) -> list:
        all_queries = []
        for file in os.listdir(input_file):
            if file.endswith(".json"):
                with open(f"{input_file}/{file}", "r", encoding="utf-8") as jsonfile:
                    data = json.load(jsonfile)
                    for item in data:
                        query = {
                            "input": item["Question"],
                            "output": {
                                "answer": item["Must_have"],
                                "provenance": [{"title": item["Question"]}],
                            },
                        }
                        all_queries.append(query)

        return all_queries

    def process_documents(
        self,
        query_file: str,
        db: DocDB,
        queries: dict = None,
        raw_query_dir: str = "data/raw/MedLFQA",
    ) -> dict:
        if queries is None:
            with open(query_file, "r", encoding="utf-8") as jsonfile:
                queries = json.load(jsonfile)

        documents = {}
        for query in queries:
            query_text = query["input"]
            docs = self._get_documents_per_query(query_text, raw_query_dir)
            documents[query_text] = docs

        return

    def _get_documents_per_query(self, query: str, raw_query_dir: str) -> list:
        """Returns a list of documents for a given query."""
        datasets = {}
        for file in os.listdir(raw_query_dir):
            if file.endswith(".json"):
                with open(f"{raw_query_dir}/{file}", "r", encoding="utf-8") as jsonfile:
                    datasets[file] = json.load(jsonfile)

        documents = {}
        for _, dataset in datasets.items():
            for pt in dataset:
                pt_docs = []
                pt_docs.extend(
                    [
                        item.strip()
                        for item in pt["Free_form_answer"].rstrip(".").split(".")
                    ]
                )
                pt_docs.extend(pt["Nice_to_have"])
                documents[pt["Question"]] = pt_docs

        try:
            docs = documents[query]
            # contents = ". ".join(docs)
            return docs
        except Exception as e:
            print(f"Error retrieving documents for query {query}: {e}")
            return []
