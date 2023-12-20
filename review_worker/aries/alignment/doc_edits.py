import collections
import difflib
import html
import io
import itertools
import logging
import re
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import nltk.corpus
import nltk.util
import numpy as np
import tqdm
from nltk.util import ngrams

import aries.util.data
import aries.util.edit
import aries.util.s2orc
from aries.util.color import colorprint
from aries.util.s2orc import load_s2orc

logger = logging.getLogger(__name__)


class ParagraphEdit:
    def __init__(self, texts1, texts2, source_idxs, target_idxs, preceding_source_idx=None, similarity=None, doc_edits=None, edit_id=None):
        self.texts1 = texts1
        self.texts2 = texts2
        self.doc_edits = doc_edits
        self.edit_id = edit_id
        self.preceding_sidx = preceding_source_idx
        self.source_idxs = source_idxs if source_idxs is not None else []
        self.target_idxs = target_idxs if target_idxs is not None else []
        self.similarity = similarity

        if self.preceding_sidx is not None and len(self.source_idxs) != 0:
            raise ValueError(
                "Edit cannot be both an addition and a revision, but got both preceding idx {} and source idxs {}".format(
                    self.preceding_sidx, str(self.source_idxs)
                )
            )

        self._diff = None

    def get_source_text(self):
        return "\n".join([self.texts1[idx] for idx in self.source_idxs])

    def get_target_text(self):
        return "\n".join([self.texts2[idx] for idx in self.target_idxs])

    def get_diff(self):
        if self._diff is None:
            self._diff = list(difflib.ndiff(self.get_source_text().split(), self.get_target_text().split()))
        return self._diff

    def print_diff(self, color_format="ansi", **kwargs):
        t1 = self.get_source_text().split(" ")
        t2 = self.get_target_text().split(" ")
        if len(t1) == 0 or len(t2) == 0:
            return

        wdiff = aries.util.edit.make_word_diff(t1, t2, color_format=color_format)
        wdiff = re.sub(r"^ ?\[([+-])", r" [\1", wdiff)
        print(wdiff, **kwargs)

    def get_added_tokens(self):
        if len(self.source_idxs) == 0:
            return self.get_target_text().split()
        return self._get_diff_tokens_by_type("+")

    def get_removed_tokens(self):
        if len(self.target_idxs) == 0:
            return self.get_source_text().split()
        return self._get_diff_tokens_by_type("-")

    def get_preserved_tokens(self):
        return self._get_diff_tokens_by_type(" ")

    def is_identical(self):
        return self.get_source_text() == self.get_target_text()

    def is_full_addition(self):
        return len(self.source_idxs) == 0

    def is_full_deletion(self):
        return len(self.target_idxs) == 0

    def _get_diff_tokens_by_type(self, type_char):
        tokens = []
        for tok in self.get_diff():
            if tok[0] == type_char:
                tokens.append(tok[2:])
        return tokens


class DocEdits:
    def __init__(self, s2orc1, s2orc2, paragraph_edits: List[ParagraphEdit] = None):
        self.s2orc1 = s2orc1
        self.s2orc2 = s2orc2
        self.paragraph_edits = paragraph_edits if paragraph_edits is not None else []
        self.source_target_map = dict()
        self.target_source_map = dict()
        self.near_edits_map = dict()
        self.edit_ids_map = dict()

        self._source_texts = [x["text"] for x in self.s2orc1["pdf_parse"]["body_text"]]
        self._target_texts = [x["text"] for x in self.s2orc2["pdf_parse"]["body_text"]]

        for ed in self.paragraph_edits:
            self.add_edit(ed)

    def add_edit(self, ed):
        # Check that edit can be safely added
        for sidx in ed.source_idxs:
            if sidx in self.source_target_map:
                raise ValueError("Edit must have unique source indexes but got conflict for {}".format(sidx))
        for tidx in ed.target_idxs:
            if tidx in self.target_source_map:
                raise ValueError("Edit must have unique target indexes but got conflict for {}".format(tidx))

        # Add edit
        self.paragraph_edits.append(ed)
        for sidx in ed.source_idxs:
            self.source_target_map[sidx] = ed
        for tidx in ed.target_idxs:
            self.target_source_map[tidx] = ed

        if ed.preceding_sidx is not None:
            if ed.preceding_sidx not in self.near_edits_map:
                self.near_edits_map[ed.preceding_sidx] = []
            self.near_edits_map[ed.preceding_sidx].append(ed)

        if ed.edit_id is not None:
            if ed.edit_id in self.edit_ids_map:
                raise ValueError("Duplicate edit id {}".format(ed.edit_id))
            self.edit_ids_map[ed.edit_id] = ed

    def make_edit(self, *args, **kwargs):
        return ParagraphEdit(self._source_texts, self._target_texts, *args, doc_edits=self, **kwargs)

    def iter_source_edits(self):
        """Iterate through paragraph edits in order of lowest source idx"""
        # Go through paras in order, but make sure newly-added paras go after the preceding para
        estimated_source_idxs = []
        edits = sorted(self.paragraph_edits, key=lambda x: min(x.target_idxs) if len(x.target_idxs) != 0 else float("inf"))
        for edit in edits:
            if len(edit.source_idxs) != 0:
                estimated_source_idxs.append((min(edit.source_idxs), 0))
            elif edit.preceding_sidx is not None:
                estimated_source_idxs.append((edit.preceding_sidx + 0.5, 0))
            else:
                last_esi = estimated_source_idxs[-1] if len(estimated_source_idxs) != 0 else (-1, 0)
                estimated_source_idxs.append((last_esi[0], last_esi[1] + 1))
        for est_idx, edit in sorted(zip(estimated_source_idxs, edits), key=lambda x: x[0]):
            yield edit

    def has_source_edit(self, sidx):
        return sidx in self.source_target_map

    def source_edit(self, sidx):
        return self.source_target_map[sidx]

    def has_target_edit(self, tidx):
        return tidx in self.target_source_map

    def target_edit(self, tidx):
        return self.target_source_map[tidx]

    def by_id(self, edit_id):
        return self.edit_ids_map[edit_id]

    def get_unmapped_source_idxs(self):
        return [sidx for sidx in range(len(self.s2orc1["pdf_parse"]["body_text"])) if sidx not in self.source_target_map]

    def get_unmapped_target_idxs(self):
        return [tidx for tidx in range(len(self.s2orc2["pdf_parse"]["body_text"])) if tidx not in self.target_source_map]

    def to_json(self):
        edits_json = []
        if any([x.edit_id is not None for x in self.paragraph_edits]):
            edits_json = [
                {
                    "edit_id": edit.edit_id,
                    "source_idxs": edit.source_idxs,
                    "target_idxs": edit.target_idxs,
                }
                for edit in self.iter_source_edits()
            ]
        else:
            edits_json = [
                {
                    "edit_id": edit_idx,
                    "source_idxs": edit.source_idxs,
                    "target_idxs": edit.target_idxs,
                }
                for edit_idx, edit in enumerate(self.iter_source_edits())
            ]
        return {"source_pdf_id": self.s2orc1["paper_id"], "target_pdf_id": self.s2orc2["paper_id"], "edits": edits_json}

    def make_paper_diff_string(
        doc_edits,
        print_ids_only=False,
        skip_identical=False,
        color_format="none",
        return_edit_ids=False,
    ):
        buf = io.StringIO()
        print_kwargs = {"file": buf}

        escape_fn = html.escape if color_format == "html" else lambda x: x

        if color_format == "html":
            print("<p>", end="", **print_kwargs)
        aries.util.edit.print_word_diff(doc_edits.s2orc1["abstract"], doc_edits.s2orc2["abstract"], color_format=color_format, **print_kwargs)
        print("[abstract]", **print_kwargs)
        print("edit id: 9999", **print_kwargs)
        if color_format == "html":
            print("</p>", end="", **print_kwargs)
        else:
            print(**print_kwargs)

        edits_by_id = dict()

        for edit_idx, edit in enumerate(doc_edits.iter_source_edits()):
            if (edit.is_identical() or len(edit.get_added_tokens()) == 0) and skip_identical:
                # print("skip", edit_idx)
                continue

            section_name = ""
            if len(edit.target_idxs) != 0:
                section_name = doc_edits.s2orc2["pdf_parse"]["body_text"][edit.target_idxs[0]]["section"]
            elif len(edit.source_idxs) != 0:
                section_name = doc_edits.s2orc1["pdf_parse"]["body_text"][edit.source_idxs[0]]["section"]

            if color_format == "html":
                print("<p>", end="", **print_kwargs)
            if edit.is_full_addition():
                colorprint("[+" + escape_fn(edit.get_target_text()) + "+]", color="green", form=color_format, **print_kwargs)
                if not print_ids_only:
                    print(edit.preceding_sidx, "(added)", edit.target_idxs, end="", **print_kwargs)
            elif edit.is_full_deletion():
                colorprint("[-" + escape_fn(edit.get_source_text()) + "-]", color="red", form=color_format, **print_kwargs)
                if not print_ids_only:
                    print(edit.source_idxs, "(deleted)", end="", **print_kwargs)
            else:
                edit.print_diff(color_format=color_format, **print_kwargs)
                if not print_ids_only:
                    print(edit.source_idxs, edit.target_idxs, txtcmp(edit.get_source_text(), edit.get_target_text()), end="", **print_kwargs)
            if not print_ids_only:
                print(**print_kwargs)
            print("section: {}".format(section_name or "unknown"), **print_kwargs)
            print("edit id: {}".format(edit_idx), **print_kwargs)
            edits_by_id[edit_idx] = edit

            if color_format == "html":
                print("</p>", end="", **print_kwargs)
            else:
                print(**print_kwargs)

        buf.seek(0)
        s = buf.read()
        if color_format == "html":
            s = s.replace("\n", "<br>")

        if return_edit_ids:
            return s, edits_by_id
        return s

    @staticmethod
    def from_list(s2orc1, s2orc2, edits_list):
        edits = DocEdits(s2orc1, s2orc2)
        for edit_rec in sorted(edits_list, key=lambda x: x["edit_id"]):
            edit = edits.make_edit(edit_rec["source_idxs"], edit_rec["target_idxs"], edit_id=edit_rec["edit_id"])
            edits.add_edit(edit)
        return edits


def iter_s2orc_pairs(config, doc_ids, docid2allpdfids, docid2pdfid):
    with aries.util.s2orc.S2orcFetcherSqlite(
        config.get("s2orc_db_path", ":memory:"),
        fallback_fetcher=aries.util.s2orc.S2orcFetcherFilesystem(config["s2orc_base_path"]) if config.get("s2orc_base_path", None) else None,
        update_db=False,
    ) as fetcher:
        for doc_id in tqdm.tqdm(doc_ids, desc="loading papers"):
            pdf_ids = docid2allpdfids[doc_id]
            main_pdf_id = docid2pdfid[doc_id][0]
            if main_pdf_id not in pdf_ids:
                logger.error("main pdf id {} not in pdf ids {}".format(main_pdf_id, pdf_ids))
                continue

            # id 0 is the newest one
            revised_pdf_id = pdf_ids[0]
            if revised_pdf_id == main_pdf_id:
                continue

            if not all([fetcher.has(pdf_id) for pdf_id in [main_pdf_id, revised_pdf_id]]):
                logger.warning("missing pdf ids for doc {}".format(doc_id))
                continue

            s2orc1 = load_s2orc(main_pdf_id, fetcher)

            s2orc2 = load_s2orc(revised_pdf_id, fetcher)

            yield doc_id, s2orc1, s2orc2


def txtcmp(txt1, txt2, txt1_bigram_counter=None):
    if txt1 == txt2:
        return 1
    ng1 = txt1_bigram_counter
    if txt1_bigram_counter is None:
        ng1 = collections.Counter(ngrams(txt1.split(), 2))
    ng2 = collections.Counter(ngrams(txt2.split(), 2))
    if len(ng1) == 0 and len(ng2) == 0:
        return aries.util.data.counter_jaccard(collections.Counter(txt1), collections.Counter(txt2))
    return aries.util.data.counter_jaccard(ng1, ng2)


def make_aligns(s2orc1, s2orc2, shortcut_threshold=0.4, min_threshold=0.1, window_size=30):
    aligns = dict()
    cur_offset = 0
    prev_cur_offset = 0
    for idx1, rec1 in enumerate(s2orc1["pdf_parse"]["body_text"]):
        # if rec1['text'] == 'EQUATION':
        #    continue
        best_score = 0
        # Include a window around the current estimate and also around the raw value (in case the cur_offset falls out of alignment somehow)
        idx_range = set(
            range(idx1 + cur_offset - window_size, idx1 + cur_offset + window_size)
        )  # | set(range(idx1 + cur_offset - window_size, idx1 + cur_offset + window_size))
        idx_range = sorted((x for x in idx_range if 0 <= x and x < len(s2orc2["pdf_parse"]["body_text"])), key=lambda x: abs(x - (idx1 + cur_offset)))

        # First check if there are any exact matches in the window; this is a fast test and guarantees we won't miss perfect alignments
        for idx2 in idx_range:
            rec2 = s2orc2["pdf_parse"]["body_text"][idx2]

            val = 0
            if rec1["text"] == rec2["text"]:
                val = 2
                val -= abs(idx1 + cur_offset - idx2) / len(s2orc1["pdf_parse"]["body_text"])

            if val > best_score:
                best_score = val
                aligns[idx1] = (idx2, val)

        if best_score > 1:
            prev_cur_offset = cur_offset
            cur_offset = aligns[idx1][0] - idx1
            continue

        # IF we didn't get an exact match, do the more expensive checks
        ng1 = collections.Counter(ngrams(rec1["text"].split(), 2))
        for idx2 in idx_range:
            rec2 = s2orc2["pdf_parse"]["body_text"][idx2]

            val = txtcmp(rec1["text"], rec2["text"], txt1_bigram_counter=ng1)
            val -= abs(idx1 + cur_offset - idx2) / len(s2orc1["pdf_parse"]["body_text"])

            if val > best_score:
                best_score = val
                aligns[idx1] = (idx2, val)
                if best_score > shortcut_threshold and best_score < 1.0:
                    break

        if best_score < min_threshold and idx1 in aligns:
            del aligns[idx1]

        if idx1 in aligns:
            prev_cur_offset = cur_offset
            cur_offset = aligns[idx1][0] - idx1

    return aligns


def _should_merge_edit_pair(edit1, edit2):
    # For now, we require one of the edits to be a full addition or full
    # deletion, since otherwise the corner cases get complicated
    if not (edit1.is_full_addition() or edit1.is_full_deletion() or edit2.is_full_addition() or edit2.is_full_deletion()):
        return False

    if (edit1.is_full_addition() or edit1.is_full_deletion()) and (edit2.is_full_addition() or edit2.is_full_deletion()):
        return False

    # One of these should have similarity=None in theory.
    sim_threshold = max(edit1.similarity or 0, edit2.similarity or 0)

    new_source_idxs = edit1.source_idxs.copy() + edit2.source_idxs
    new_txt1 = "".join([edit1.texts1[i] for i in new_source_idxs])
    new_target_idxs = edit1.target_idxs.copy() + edit2.target_idxs
    new_txt2 = "".join([edit1.texts2[i] for i in new_target_idxs])
    if txtcmp(new_txt1, new_txt2) > sim_threshold:
        return True
    return False


def _make_merged_edit(edit1, edit2, docedits):
    new_source_idxs = edit1.source_idxs.copy() + edit2.source_idxs
    new_target_idxs = edit1.target_idxs.copy() + edit2.target_idxs

    new_txt1 = "".join([edit1.texts1[i] for i in new_source_idxs])
    new_txt2 = "".join([edit1.texts2[i] for i in new_target_idxs])

    similarity = txtcmp(new_txt1, new_txt2)

    preceding_sidx = None
    if edit1.preceding_sidx == edit2.preceding_sidx:
        preceding_sidx = edit1.preceding_sidx

    new_edit = docedits.make_edit(new_source_idxs, new_target_idxs, similarity=similarity, preceding_source_idx=preceding_sidx)

    return new_edit


def _adjust_bad_merges(aligns):
    # We want to check if some paragraph has been split differently in the
    # different s2orcs.  So, if we could join two source paras to better align
    # to a target para, or join two target paras to better align a source para,
    # we should do that.

    # TODO: We could be much fancier with the algoritm here to handle
    # already-merged things and edited-but-also-split stuff; for now we do
    # a very basic check for easy corrections, which generally catches cases
    # where identical paras get split differently

    new_aligns = DocEdits(aligns.s2orc1, aligns.s2orc2)

    # Because we might need to merge both forwards and backwards, we do
    # in-place merges in the list of edits rather than building incrementally
    new_edit_list = sorted(aligns.paragraph_edits, key=lambda x: min(x.source_idxs) if len(x.source_idxs) != 0 else x.preceding_sidx + 0.5)

    edit_idx = 0
    while edit_idx < len(new_edit_list):
        edit = new_edit_list[edit_idx]
        if edit.is_identical() or edit.is_full_addition() or edit.is_full_deletion():
            edit_idx += 1
            continue

        # We have a partial edit, so we need to check if we can merge with preceding or following paras
        prev_edit_idx = edit_idx - 1
        while prev_edit_idx >= 0:
            prev_edit = new_edit_list[prev_edit_idx]
            if _should_merge_edit_pair(prev_edit, edit):
                logger.debug("merging %s %s %s %s", edit.source_idxs, edit.target_idxs, prev_edit.source_idxs, prev_edit.target_idxs)
                edit = _make_merged_edit(prev_edit, edit, new_aligns)
                prev_edit_idx -= 1
            else:
                break
        new_edit_list[prev_edit_idx + 1 : edit_idx + 1] = [edit]
        edit_idx = prev_edit_idx + 1

        next_edit_idx = edit_idx + 1
        while next_edit_idx < len(new_edit_list):
            next_edit = new_edit_list[next_edit_idx]
            if _should_merge_edit_pair(edit, next_edit):
                logger.debug("merging %s %s %s %s", edit.source_idxs, edit.target_idxs, next_edit.source_idxs, next_edit.target_idxs)
                edit = _make_merged_edit(edit, next_edit, new_aligns)
                next_edit_idx += 1
            else:
                break
        new_edit_list[edit_idx:next_edit_idx] = [edit]

        edit_idx += 1

    for edit in new_edit_list:
        new_aligns.add_edit(edit)

    return new_aligns


def make_full_aligns(s2orc1, s2orc2):
    aligns = make_full_aligns_v1(s2orc1, s2orc2)
    aligns = _adjust_bad_merges(aligns)

    for edit_idx, edit in enumerate(aligns.iter_source_edits()):
        assert edit.edit_id is None
        edit.edit_id = edit_idx

    return aligns


def make_full_aligns_v1(s2orc1, s2orc2):
    tmp_aligns = make_aligns(s2orc1, s2orc2)

    revmap = {v[0]: [] for k, v in tmp_aligns.items()}
    for k, v in tmp_aligns.items():
        revmap[v[0]].append(k)
    aligns = DocEdits(s2orc1, s2orc2)
    for k, v in tmp_aligns.items():
        if aligns.has_source_edit(k):
            continue
        aligns.add_edit(aligns.make_edit([kk for kk in revmap[v[0]]], [v[0]], similarity=v[1]))

    for sidx in aligns.get_unmapped_source_idxs():
        aligns.add_edit(aligns.make_edit([sidx], []))

    for idx2 in aligns.get_unmapped_target_idxs():
        tmpidx = idx2 - 1
        while tmpidx >= 0 and not aligns.has_target_edit(tmpidx):
            tmpidx -= 1
        if tmpidx < 0:
            nearidx = 0
        else:
            x = aligns.target_edit(tmpidx)
            if len(x.source_idxs) != 0:
                nearidx = max(x.source_idxs)
            elif x.preceding_sidx is not None:
                nearidx = x.preceding_sidx
            else:
                raise ValueError("Invalid mapping")
        aligns.add_edit(aligns.make_edit([], [idx2], preceding_source_idx=nearidx))

    return aligns
