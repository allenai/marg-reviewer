import json
import logging
import os
import sqlite3
import sys

import tqdm

logger = logging.getLogger(__name__)


def fuse_back_matter(s2json):
    """Fuse back matter into body text (mutating the input object) and return
    the mutated s2orc object.  Often the parser puts whatever is on the last
    pdf page into back matter even if it is clearly part of the appendix, so
    this function tries to fix that."""

    s2json["pdf_parse"]["body_text"] = s2json["pdf_parse"]["body_text"] + s2json["pdf_parse"]["back_matter"]
    return s2json


def load_s2orc(pdf_id, fetcher):
    s = fetcher.get(pdf_id)
    if s is None:
        return None
    return fuse_back_matter(s)


def iter_s2orc_pairs(base_path, paper_records, error_on_missing=True):
    with S2orcFetcherFilesystem(base_path) as fetcher:
        for record in tqdm.tqdm(paper_records, desc="loading papers"):
            doc_id = record["doc_id"]

            if not all(fetcher.has(pdf_id) for pdf_id in [record["source_pdf_id"], record["target_pdf_id"]]):
                if error_on_missing:
                    raise RuntimeError("missing pdf ids for doc {} ({}, {})".format(doc_id, record["source_pdf_id"], record["target_pdf_id"]))
                else:
                    logger.warning("missing pdf ids for doc {} ({}, {})".format(doc_id, record["source_pdf_id"], record["target_pdf_id"]))
                    continue

            s2orc1 = load_s2orc(record["source_pdf_id"], fetcher)

            s2orc2 = load_s2orc(record["target_pdf_id"], fetcher)

            yield doc_id, s2orc1, s2orc2


def iter_s2orc_docs(config, pdf_ids):
    with S2orcFetcherSqlite(
        config.get("s2orc_db_path", ":memory:"),
        fallback_fetcher=S2orcFetcherFilesystem(config["s2orc_base_path"]) if config.get("s2orc_base_path", None) else None,
        update_db=False,
    ) as fetcher:
        for pdf_id in tqdm.tqdm(pdf_ids, desc="loading papers"):
            if not fetcher.has(pdf_id):
                logger.warning("missing pdf ids for doc {}".format(pdf_id))
                continue

            s2orc2 = load_s2orc(pdf_id, fetcher)

            yield pdf_id, s2orc2


class S2orcFetcher:
    def get(self, pdf_id):
        raise NotImplementedError()

    def has(self, pdf_id):
        raise NotImplementedError()


class S2orcFetcherDummy(S2orcFetcher):
    def get(self, pdf_id):
        return None

    def has(self, pdf_id):
        return False


class S2orcFetcherSqlite(S2orcFetcher):
    def __init__(self, s2orc_db_path, fallback_fetcher=None, update_db=False):
        self.s2orc_db_path = s2orc_db_path
        self.fallback_fetcher = fallback_fetcher or S2orcFetcherDummy()
        self.update_db = update_db
        self.db = None
        self.cur = None

    def __enter__(self):
        self.db = sqlite3.connect(self.s2orc_db_path)
        self.cur = self.db.cursor()
        self.cur.execute("BEGIN")
        # We create the table/index regardless of update_db, since otherwise we hit errors later
        self.cur.execute("CREATE TABLE IF NOT EXISTS pdf_records (pdf_id TEXT PRIMARY KEY NOT NULL, title TEXT, json TEXT)")
        self.cur.execute("CREATE INDEX IF NOT EXISTS pdf_records_by_id ON pdf_records (pdf_id)")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.db.commit()
        self.db.close()

    def get(self, pdf_id):
        rec = self.cur.execute("SELECT json FROM pdf_records WHERE pdf_id=?", (pdf_id,)).fetchone()
        if rec is not None:
            return json.loads(rec[0])
        s2orc_json = self.fallback_fetcher.get(pdf_id)

        if self.update_db and s2orc_json is not None:
            self.cur.execute("INSERT INTO pdf_records (pdf_id, title, json) VALUES (?, ?, ?)", (pdf_id, s2orc_json["title"], json.dumps(s2orc_json)))
        return s2orc_json

    def has(self, pdf_id):
        rec = self.cur.execute("SELECT 1 FROM pdf_records WHERE pdf_id=?", (pdf_id,)).fetchone()
        if rec is not None:
            return True
        return self.fallback_fetcher.has(pdf_id)


class S2orcFetcherFilesystem(S2orcFetcher):
    def __init__(self, s2orc_base_path):
        self.s2orc_base_path = s2orc_base_path

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return

    def get(self, pdf_id):
        if not self.s2orc_base_path:
            return None

        path = os.path.join(self.s2orc_base_path, "{}.json".format(pdf_id))

        try:
            with open(path) as f:
                return json.load(f)
        except FileNotFoundError:
            return None

    def has(self, pdf_id):
        if not self.s2orc_base_path:
            return False
        path = os.path.join(self.s2orc_base_path, "{}.json".format(pdf_id))
        return os.path.exists(path)
