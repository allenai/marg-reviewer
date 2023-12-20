import collections
import difflib
import itertools
from typing import Iterable, List, Tuple, Union

import numpy as np
import tqdm
from cffi import FFI

from .color import colorify, colorprint


def init_levenshtein_c():
    ffibuilder = FFI()
    ffibuilder.set_source(
        "_levenshtein",
        r"""
            int levenshtein(int *seq1, int seq1_len, int *seq2, int seq2_len, int *v0)
            {
                // Adapted from https://en.wikipedia.org/wiki/Levenshtein_distance  (CC-BY-SA)

                // v0 is just a buffer for temporary calculations; easier to
                // ask the caller to allocate it than to deal with C mem
                // management

                int substitutionCost, insertionCost, deletionCost;
                int tmpval;

                for (int i = 0; i < seq2_len+1; i++) {
                    v0[i] = i;
                }

                for (int i = 0; i < seq1_len; i++) {
                    // calculate v1 (current row distances) from the previous row v0

                    // first element of v1 is A[i+1][0]
                    //   edit distance is delete (i+1) chars from s to match empty t
                    tmpval = i + 1;

                    // use formula to fill in the rest of the row
                    for(int j = 0; j < seq2_len; j++) {
                        // calculating costs for A[i+1][j+1]
                        deletionCost = v0[j + 1] + 1;
                        insertionCost = tmpval + 1;
                        substitutionCost = v0[j];
                        if (seq1[i] != seq2[j]) {
                            substitutionCost++;
                        }

                        v0[j] = tmpval;

                        tmpval = deletionCost;
                        if (insertionCost < tmpval) {
                            tmpval = insertionCost;
                        }
                        if (substitutionCost < tmpval) {
                            tmpval = substitutionCost;
                        }
                    }
                    v0[seq2_len] = tmpval;
                }
                // after the last swap, the results of v1 are now in v0
                return v0[seq2_len];
            }
        """,
    )

    ffibuilder.cdef("int levenshtein(int*, int, int*, int, int*);")

    # Compile the C module and import it
    ffibuilder.compile(verbose=True)
    from _levenshtein import ffi, lib

    return ffi, lib


levenshtein_ffi, levenshtein_lib = None, None


def levenshtein_distance(seq1, seq2):
    # We call a C function for levenshtein via CFFI because it is about 1000x
    # faster than the python version (the difference between running in an hour
    # vs running in a month)

    global levenshtein_ffi, levenshtein_lib

    if levenshtein_ffi is None:
        levenshtein_ffi, levenshtein_lib = init_levenshtein_c()

    if isinstance(seq1, str):
        seq1 = [ord(c) for c in seq1]

    if isinstance(seq2, str):
        seq2 = [ord(c) for c in seq2]

    if len(seq1) > len(seq2):
        seq1, seq2 = seq2, seq1

    # Important: these arrs need to be in their own variables, NOT inlined with
    # the levenshtein_ffi.from_buffer, or else the GC will free the memory and
    # memory will get corrupted (often manifests as seq2 overwriting seq1, but
    # also can segfault)
    seq1_arr = np.array(seq1, dtype=np.int32)
    seq2_arr = np.array(seq2, dtype=np.int32)
    v0_arr = np.zeros(len(seq2) + 1, dtype=np.int32)

    seq1_buf = levenshtein_ffi.cast("int*", levenshtein_ffi.from_buffer(seq1_arr))
    seq2_buf = levenshtein_ffi.cast("int*", levenshtein_ffi.from_buffer(seq2_arr))
    v0 = levenshtein_ffi.cast("int*", levenshtein_ffi.from_buffer(v0_arr))

    result = levenshtein_lib.levenshtein(seq1_buf, len(seq1), seq2_buf, len(seq2), v0)
    return result


def basic_token_align(seq1, seq2, seq2_ignored_ids: Iterable = None):
    """Aligns the tokens of seq1 and seq2 assuming that seq2 contains all the
    characters of seq1, but possibly with some extra tokens (e.g., special
    whitespace markers from a huggingface transformers tokenizer) and possibly
    partitioned differently.

    In cases where the boundaries are mismatched, this maps to the token with
    largest overlap, and breaks ties in favor of earlier tokens.

    if seq2_ignored_ids is given, the specified token indexes in seq2 are
    ignored and will not be aligned to anything in seq1.

    Returns a tuple (dist, alignment) where dist is the total of mismatches
    (number of characters that seq2 token boundaries had to be moved to
    complete alignment) and `alignment` is a list of the same length as seq2
    containing the indexes of the aligned tokens from seq1 (or None if the
    token did not overlap seq1 at all)."""

    if seq2_ignored_ids is None:
        seq2_ignored_ids = set()

    # if seq1[0] == 'numerous':
    #    breakpoint()

    seq1idxs = list(itertools.chain(*[[(idx, c) for c in tok] for idx, tok in enumerate(seq1)]))
    seq2idxs = list(itertools.chain(*[[(idx, c) for c in tok] for idx, tok in enumerate(seq2)]))

    seq2_seq1_char_align = [None] * len(seq2idxs)
    idx1 = 0
    last_valid = None
    for chridx2, (idx2, c2) in enumerate(seq2idxs):
        if idx1 >= len(seq1idxs):
            break
        if c2 == seq1idxs[idx1][1] and idx2 not in seq2_ignored_ids:
            seq2_seq1_char_align[chridx2] = idx1
            last_valid = idx1
            idx1 += 1

    # Ensure that all chars of seq1 were mapped to a char in seq2
    # if ''.join(seq1) != ''.join(seq2):
    if last_valid != (len(seq1idxs) - 1):
        raise ValueError("Cannot align: Sequences didn't contain the same characters")

    # Align the sequences
    alignment_counts = {idx: collections.Counter() for idx in range(len(seq2))}
    # for idx1, idx2 in zip(seq1idxs, seq2idxs):
    for chridx1, (idx2, c2) in zip(seq2_seq1_char_align, seq2idxs):
        idx1 = seq1idxs[chridx1][0] if chridx1 is not None else None
        alignment_counts[idx2][idx1] += 1

    alignments = []
    n_mismatch_total = 0
    for idx2 in range(len(seq2)):
        best_idxs = sorted(
            alignment_counts[idx2].keys(), reverse=True, key=lambda x: (alignment_counts[idx2][x], -x if x is not None else float("-inf"))
        )
        best_idx1 = best_idxs[0]
        if best_idx1 is None and len(best_idxs) > 1:
            best_idx1 = best_idxs[1]
        n_mismatch_total += sum(alignment_counts[idx2].values()) - alignment_counts[idx2][best_idx1]
        alignments.append(best_idx1)

    return (n_mismatch_total, alignments)


def print_word_diff(text1, text2, color_format="ansi", **print_kwargs):
    print(make_word_diff(text1, text2, color_format=color_format), **print_kwargs)


def make_word_diff(text1, text2, color_format="ansi"):
    if not isinstance(text1, list):
        text1 = text1.split(" ") if len(text1) != 0 else []

    if not isinstance(text2, list):
        text2 = text2.split(" ") if len(text2) != 0 else []

    prevtok = " "
    parity = 0

    def color_for_tok(tok):
        if color_format == "none":
            return None

        if tok == "+":
            return "green"
        elif tok == "-":
            return "red"
        elif tok == "?":
            return "blue"
        return None

    s = ""
    for idx, x in enumerate(difflib.ndiff(text1, text2)):
        if prevtok != x[0] and prevtok in ("+", "-"):
            s += colorify(prevtok + "]", color=color_for_tok(prevtok), form=color_format)
        if prevtok != x[0] and x[0] in ("+", "-"):
            if parity == 0 and idx > 0:
                s += " "
            s += colorify("[" + x[0], color=color_for_tok(x[0]), form=color_format)

        if x[0] == " ":
            if idx != 0:
                s += " "
            s += x[2:]
            parity = 0
        elif x[0] == "?":
            pass
        else:
            # s = '['+x[0]+x[1:]+x[0]+']'
            if prevtok != x[0]:
                parity = parity ^ 1
            else:
                s += " "
            s += colorify(x[2:], color=color_for_tok(x[0]), form=color_format)
        prevtok = x[0]

    if prevtok in ("+", "-"):
        s += colorify(prevtok + "]", color=color_for_tok(prevtok), form=color_format)

    return s


def build_offsets(
    toks: Union[str, List[str]],
    chunk_length: int,
) -> dict:
    offsets = dict()
    for idx in range(len(toks) - chunk_length + 1):
        chunk = tuple(toks[idx : idx + chunk_length])
        if chunk not in offsets:
            offsets[chunk] = []
        offsets[chunk].append(idx)
    return offsets


def update_overlaps(
    cur_overlaps: List[Tuple[int, int]],
    toks1: Union[str, List[str]],
    toks2: Union[str, List[str]],
    idx2: int,
    min_length: int,
) -> Tuple[List[Tuple[int, int]], List[Tuple[Tuple[int, int], Tuple[int, int]]]]:
    overlaps = []
    new_overlaps = []
    for overlap in cur_overlaps:
        overlap_length = idx2 - overlap[1]
        end1 = overlap[0] + overlap_length
        if end1 < len(toks1) and idx2 < len(toks2) and toks1[end1] == toks2[idx2]:
            new_overlaps.append(overlap)
        elif overlap_length >= min_length:
            overlaps.append(((overlap[0], overlap[0] + overlap_length), (overlap[1], overlap[1] + overlap_length)))
    return new_overlaps, overlaps


def find_overlapping_substrings(
    toks1: Union[str, List[str]],
    toks2: Union[str, List[str]],
    min_length: int = 32,
):
    """
    Finds overlapping substrings of toks1 and toks2, where toks1 and toks2 are
    lists of tokens.

    min_length is the minimum number of tokens that a match must span in order
    to be returned

    Returns a list of pairs of spans, e.g. [((10, 20), (14, 24))].  Each span
    pair is a (start_idx, end_idx) tuple representing a half-open interval.

    Any long match technically contains many shorter matches.  This function
    returns only the longest match for each set; for each returned pair of
    spans (span1, span2), there will be no other returned pair (span3, span4)
    such that span3 contains span1 AND span4 contains span2.
    """
    if len(toks1) == 0 or len(toks2) == 0:
        return []

    # Use chunks to reduce number of hits per token, but don't go too high
    # since mem usage is len(toks1)*chunk_length. If character tokenization and
    # long chunk_length (e.g., 1000), then we would use 1000x the memory needed
    # to store toks1.
    chunk_length = min(min_length, 10)
    offsets1 = build_offsets(toks1, chunk_length)
    overlaps = []
    cur_overlaps = []

    for idx2, tk2 in enumerate(toks2):
        cur_overlaps, new_overlaps = update_overlaps(cur_overlaps, toks1, toks2, idx2, min_length)
        overlaps.extend(new_overlaps)

        if idx2 <= (len(toks2) - min_length):
            chunk = tuple(toks2[idx2 : idx2 + chunk_length])
            for idx1 in offsets1.get(chunk, []):
                has_overlap = False
                for overlap in cur_overlaps:
                    overlap_length = idx2 - overlap[1]
                    if idx1 - overlap_length == overlap[0]:
                        has_overlap = True
                        break
                if not has_overlap:
                    cur_overlaps.append((idx1, idx2))

    idx2 = len(toks2)
    _, new_overlaps = update_overlaps(cur_overlaps, toks1, toks2, idx2, min_length)
    overlaps.extend(new_overlaps)

    final_overlaps = []
    for o1 in overlaps:
        is_subset = False
        for o2 in overlaps:
            if o1 != o2 and o1[0][0] >= o2[0][0] and o1[0][1] <= o2[0][1] and o1[1][0] >= o2[1][0] and o1[1][1] <= o2[1][1]:
                is_subset = True
                break
        if not is_subset:
            final_overlaps.append(o1)
    return final_overlaps
