import collections
import functools
import itertools
import json
import logging
import os
import sys

import numpy as np
import tqdm

if os.path.abspath("../..") not in sys.path:
    sys.path.insert(0, os.path.abspath("../.."))

import aries
from aries.alignment.doc_edits import DocEdits
from aries.review_generation.multi_agent import GptChatBot, make_chunked_paper_diff
from aries.util.color import colorprint
from aries.util.data import index_by, iter_jsonl_files
from aries.util.logging import init_logging
from run_reviewgen import make_review

logger = logging.getLogger(__name__)

DEFAULT_GPT_ALIGN_PROMPTS = {
    "align_comment_lists_system_prompt_set1": 'Instructions:\nYou must examine two lists of review comments and identify pairs of comments that have the same meaning.\n\nA user will provide two lists of comments in JSON format; one with "reference" comments (from real reviewers) and one with "predicted" comments.  Each comment will be associated with an ID number that can be used to identify it.  You must find pairs of comments between the two lists that have the same meaning or make essentially the same request.  For each pair, you must determine whether the predicted comment has "less", "more", or the "same" level of specificity.\n\nYou can think step-by-step and write out your reasoning as needed in order to create a good list.  However, when you are finished thinking, you should write "Result:" on a new line, followed by the a JSON list where each object has a "reference_id", a "predicted_id", and a "relative_specificity".\n\nFor example, suppose the reference list is `[{"comment_id": 54, "comment": "Compare your method with BERT"}, {"comment_id": 56, "comment": "Report standard deviations in tables"}]` and the predicted list is `[{"comment_id": 25, "comment": "Compare with more baselines"}, {"comment_id": 56, "comment": "Paper lacks novelty"}]`.  In this case, there is one matching pair (comparing to baselines/BERT), and asking for more baselines is less specific than asking for BERT, so the final output would be `Result: [{"reference_id": 54, "predicted_id": 25, "relative_specificity": "less"}]`.  Notice that there can be reference or predicted comments that do not align to anything.  In some cases there could be no alignments, in which case you should output an empty list.  There may also be cases where two (or more) reference or predicted comments are similar and therefore both align to the same comment in the other list.',
    "align_comment_lists_system_prompt_set2": 'Instructions:\nYou must carefully consider two review comments that were made about a scientific paper.  One is the "reference" comment and the other is the "predicted" comment.  Your job is to determine whether the two comments have essentially the same meaning, although they may have different levels of specificity.  For example, the comments "Not enough datasets were used for comparison" and "Please test on ImageNet" are both suggesting that the authors test their methods on more datasets, but the second one is more specific (has higher specificity) than the first because it suggests a particular dataset.  On the other hand, "Not enough datasets were used for comparison" and "Not enough models were used for comparison" don\'t have the same meaning, because one is about datasets and the other is about models; they are different critiques/suggestions.\n\nYou can think step-by-step and write out your reasoning as needed to make a good decision.  However, when you are finished thinking, you must write "Result:" on a new line, followed by a JSON object with the fields:\n- "relatedness": a string, either "none" (no relation), "weak" (the comments are slightly related, perhaps sharing a topic, and it is possible that an edit to the paper that addresses one might also address the other, but they are not asking for the same thing and there could be an edit to the paper that addresses one but not the other), "medium" (the comments are asking for roughly the same thing and it is likely that an edit addressing one would also address the other, at least to some extent; however, they may be framed differently, use a different tone, put emphasis on different factors, or differ in some details), or "high" (the comments are making almost exactly the same request and use roughly the same framing, tone, emphasis, and details)\n- "relative_specificity" (a string, "more" if the predicted comment is more specific than the reference comment, "less" if it is less specific, and "same" if it is the same specificity, or null if relatedness is "none").  If (and only if) it is absolutely impossible to determine whether the two comments have the same meaning without more information, you can set relatedness to "unknown" and add a "note" field (a string) explaining why it is impossible to determine.  However, this should be a rare occurrence.',
}


@functools.cache
def get_all_paper_edits(aries_base_path):
    return index_by(
        iter_jsonl_files([os.path.join(aries_base_path, "paper_edits.jsonl")]),
        "doc_id",
        one_to_one=True,
    )


def load_edits_for_doc(doc_id, config):
    paper_edits = get_all_paper_edits(config["aries_base_path"])[doc_id]
    # Use aries.util.s2orc loader to handle back_matter merging
    with aries.util.s2orc.S2orcFetcherFilesystem(
        os.path.join(config["aries_base_path"], "s2orc/")
    ) as fetcher:
        s2orc1 = aries.util.s2orc.load_s2orc(paper_edits["source_pdf_id"], fetcher)
        s2orc2 = aries.util.s2orc.load_s2orc(paper_edits["target_pdf_id"], fetcher)
    doc_edits = DocEdits.from_list(s2orc1, s2orc2, paper_edits["edits"])
    return doc_edits


def categorize_human_comments(review_replies_by_doc, test_docids_subset):
    extracted_comments_by_docreview = dict()
    for doc_id in test_docids_subset:
        extracted_comments_by_docreview[doc_id] = []
        for review_rec in review_replies_by_doc[doc_id]:
            reviewtext = "\n\n".join(
                [
                    v
                    for k, v in review_rec["content"].items()
                    if k
                    in {
                        "summary_of_the_paper",
                        "main_review",
                        "summary_of_the_review",
                        "review",
                    }
                ]
            )
            tmpbot = GptChatBot(
                config["gpt3_cache_db_path"],
                model="gpt-4-0613",
                system_prompt="""Instructions:\nA user will give you a scientific paper review, and you must make the list of comments made by the reviewer.  Write each specific suggestion or critique that the reviewer makes.  Each item in the list should stand alone as a complete comment, so you may need to paraphrase or adjust comments in order to add context and improve clarity.  However, you should try to preserve the original wording when possible.  In addition, you should merge similar comments as needed to ensure that each final comment in your list stands on its own as a fully-contextualized comment.  For example, a reviewer might give a high-level comment like "Experiments are not convincing" and then elaborate on that comment later with a more detailed explanation of how the experiments are unconvincing; in this case, you should merge the two comments into a single comment with all the details.

Your output should be a JSON object like `{"major": List[str], "minor": List[str]}` where the lists of strings are the lists of review comments.  The "major" comments should be the most important ones, typically regarding the impact and novelty of the work, the correctness of main claims, or anything else that the reviewer suggests is an important factor in accepting the work.  The "minor" comments should be the ones that are just about small details that aren't crucial for the work, such as style and grammar, minor clarifications, or other things that the reviewer indicates aren't important.

Example:\nSuppose a reviewer says the following:
"The paper is well organised and the problem is very well motivated.  However, I think it is missing some key things, such as experiments with larger models and standard deviations values in tables.  In addition, the work is incremental as Wang et al have already studied Pre-LN Transformers.  For example, they have shown that Pre-LN transformers are better that Post-LN transformers when the network has many transformer layers, they explain theoretically why this is the case.  Some larger models the authors could consider are BERT and T5.  Also, while the paper is well-organized, the grammar is poor.  Finally, the symbol T is used in Equation 3 but not defined until after Equation 5, which is confusing."

You might break this down in the way shown below.  Notice that the positive comment is not included, some context has been added to the reviewer's list to allow each element to stand alone, and the related comments about larger models were merged to create one fully-contextualized comment:
{
    "major": [
        "The paper is missing experiments with larger models, such as BERT and T5",
        "The paper is missing standard deviation values in tables",
        "The work is incremental as Wang et al have already studied Pre-LN in Transformers.  For example, they have shown that Pre-LN transformers are better that Post-LN transformers when the network has many transformer layers, they explain theoretically why this is the case."
    ],
    "minor": [
        "While the paper is well-organized, the grammar is poor.",
        "The symbol T is used in Equation 3 but not defined until after Equation 5, which is confusing."
    ]
}
""",
                max_tokens=4096,
            )
            try:
                text1, resp = tmpbot.chat(
                    """Review:\n\n{review}""".format(review=reviewtext)
                )
                obj = json.loads(text1)
                extracted_comments_by_docreview[doc_id].append(obj)
            except Exception:
                continue
    return extracted_comments_by_docreview


def gpt_align_comment_lists(
    ref_comments, predicted_comments, prompts=DEFAULT_GPT_ALIGN_PROMPTS
):
    new_ref_comments = [
        {"comment_id": cidx, "comment": c} for cidx, c in enumerate(ref_comments)
    ]
    new_pred_comments = [
        {"comment_id": cidx, "comment": c} for cidx, c in enumerate(predicted_comments)
    ]

    resultsets = []
    for trialnum in range(5):
        tmpbot = GptChatBot(
            config["gpt3_cache_db_path"],
            model="gpt-4-0613",
            system_prompt=prompts["align_comment_lists_system_prompt_set1"],
            max_tokens=2048,
        )
        rcs = new_ref_comments.copy()
        np.random.default_rng(trialnum).shuffle(rcs)
        pcs = new_pred_comments.copy()
        np.random.default_rng(trialnum).shuffle(pcs)
        refcomstr = "Reference comments:\n{ref_comments}".format(
            ref_comments=json.dumps(rcs, indent=4)
        )
        predcomstr = "Predicted comments:\n{pred_comments}".format(
            pred_comments=json.dumps(pcs, indent=4)
        )
        comprompt = "\n\n".join(
            [refcomstr, predcomstr] if trialnum % 2 == 0 else [predcomstr, refcomstr]
        )
        rt, resp = tmpbot.chat(comprompt)
        try:
            result = json.loads(rt[rt.find("Result:") + len("Result:") :])
        except json.decoder.JSONDecodeError:
            logger.error("JSON decoding failed in alignment; trying again...")
            rt, resp = tmpbot.chat(
                'Print "Result:" followed by the final result as JSON.  Do not include any additional commentary.'
            )
            result = json.loads(rt[rt.find("Result:") + len("Result:") :])
        # Check that ref/pred ids are valid and filter out any that aren't
        result = [
            x
            for x in result
            if int(x["reference_id"]) < len(rcs) and int(x["predicted_id"]) < len(pcs)
        ]
        resultsets.append(result)

    pair_counts = collections.Counter()
    for result in resultsets:
        for pair in result:
            pair_counts[(pair["reference_id"], pair["predicted_id"])] += 1
    logger.info("Pair counts: {}".format(pair_counts.most_common()))

    final_pairs = []
    for pair, count in pair_counts.most_common():
        if count < 2:
            continue
        # Check the specific pair with GPT to confirm the match
        tmpbot = GptChatBot(
            config["gpt3_cache_db_path"],
            model="gpt-4-0613",
            system_prompt=prompts["align_comment_lists_system_prompt_set2"],
            max_tokens=2048,
        )
        ref_comment = new_ref_comments[int(pair[0])]["comment"]
        pred_comment = new_pred_comments[int(pair[1])]["comment"]
        rt, resp = tmpbot.chat(
            "Reference comment:\n{ref_comment}\n\nPredicted comment:\n{pred_comment}".format(
                ref_comment=ref_comment, pred_comment=pred_comment
            )
        )
        print(
            {
                "reference_comment": ref_comment,
                "predicted_comment": pred_comment,
            }
        )
        colorprint(rt, color="cyan")
        result_obj = json.loads(rt[rt.find("Result:") + len("Result:") :])
        result_obj["same_meaning"] = result_obj["relatedness"] in (
            "medium",
            "high",
        )
        if result_obj["same_meaning"]:
            final_pairs.append(
                {
                    "reference_id": pair[0],
                    "predicted_id": pair[1],
                    "relative_specificity": result_obj["relative_specificity"],
                    "relatedness": result_obj["relatedness"],
                    "pairwise_result": result_obj,
                    "count": count,
                    "score": count / len(resultsets),
                }
            )
        elif result_obj["same_meaning"] is None:
            logger.info(
                "FAILED TO DETERMINE SAME_MEANING on {}: {}".format(
                    json.dumps(pair), json.dumps(result_obj)
                )
            )

    result = final_pairs

    for pair in result:
        pair["reference_comment"] = new_ref_comments[int(pair["reference_id"])][
            "comment"
        ]
        pair["predicted_comment"] = new_pred_comments[int(pair["predicted_id"])][
            "comment"
        ]
    return result


def make_reviews_for_doc(doc_id, config):
    doc_edits = load_edits_for_doc(doc_id, config)

    with aries.util.gpt3.Gpt3CacheClient(config["gpt3_cache_db_path"]) as gptcli:
        paper_chunks, _ = make_chunked_paper_diff(
            doc_edits,
            config["paper_chunk_size"],
            gptcli,
            mode="source",
            model_name="gpt-4",
            color_format="none",
            doc_mods={"delete_ids": [], "paragraph_modifications": []},
        )
    return make_review(config, paper_chunks, doc_edits.s2orc1, doc_id=doc_id)


def main(config):
    global all_paper_edits
    all_paper_edits = index_by(
        iter_jsonl_files(
            [os.path.join(config["aries_base_path"], "paper_edits.jsonl")]
        ),
        "doc_id",
        one_to_one=True,
    )

    comments_by_docid = index_by(
        iter_jsonl_files(
            [os.path.join(config["aries_base_path"], "review_comments.jsonl")]
        ),
        "doc_id",
    )
    split_ids = list(
        iter_jsonl_files([os.path.join(config["aries_base_path"], "split_ids.json")])
    )[0]
    review_replies_by_doc = index_by(
        iter_jsonl_files(
            [os.path.join(config["aries_base_path"], "review_replies.jsonl")]
        ),
        "forum",
    )

    # For speed/cost, just eval a subset
    N_DOCS = 30
    test_docids_subset = [
        x["doc_id"]
        for x in [
            xx for xx in split_ids["test"] if len(comments_by_docid[xx["doc_id"]]) != 0
        ][:N_DOCS]
    ]
    list(itertools.chain(*[comments_by_docid[x] for x in test_docids_subset]))

    extracted_comments_by_docreview = categorize_human_comments(
        review_replies_by_doc,
        test_docids_subset,
    )

    generated_comments_by_doc = None
    if config["generated_comments_file"] and os.path.exists(
        config["generated_comments_file"]
    ):
        with open(config["generated_comments_file"]) as f:
            generated_comments_by_doc = index_by(
                json.load(f), "doc_id", one_to_one=True
            )

    all_metrics = []
    all_results = []
    all_aligns = []
    for doc_id in tqdm.tqdm(test_docids_subset):
        if generated_comments_by_doc:
            generated_comments = generated_comments_by_doc[doc_id]["generated_comments"]
        else:
            rev1 = make_reviews_for_doc(doc_id, config)
            generated_comments = []
            try:
                generated_comments = rev1 or []
            except json.decoder.JSONDecodeError as e:
                logger.exception("Error on {} ({})".format(didx, doc_id))

        # Generated comments is either a list, or a dict with comments per sub-category
        if isinstance(generated_comments, list):
            generated_comments = {"all": generated_comments}

        if "all" not in generated_comments:
            generated_comments["all"] = list(
                itertools.chain(*generated_comments.values())
            )

        logger.info(
            "FINAL REVIEW:\n{}".format(
                "\n".join(
                    [
                        ("- " + x if not x.startswith("-") else x)
                        for x in generated_comments["all"]
                    ]
                )
            )
        )

        aligns_for_reviews = {k: [] for k in generated_comments.keys()}
        metrics_for_reviews = {k: [] for k in generated_comments.keys()}
        ref_comment_sets = []

        # Get alignments from our generated comments against each human-written review
        reviews_to_iter = extracted_comments_by_docreview[doc_id]
        for dr_comments in reviews_to_iter:
            reference_comments = dr_comments["major"]
            ref_comment_sets.append(reference_comments)

            aligns_out = gpt_align_comment_lists(
                reference_comments, generated_comments["all"]
            )
            aligns_out = [x for x in aligns_out if x["relative_specificity"] != "less"]

            for k, gen_comments in generated_comments.items():
                subreview_aligns = [
                    x for x in aligns_out if x["predicted_comment"] in gen_comments
                ]
                aligns_for_reviews[k].append(subreview_aligns)

                logger.info(
                    "Alignment output ({k}):\n{aligns}".format(
                        k=k, aligns=json.dumps(subreview_aligns, indent=4)
                    )
                )

                refids = set(x["reference_id"] for x in subreview_aligns)
                predids = set(x["predicted_id"] for x in subreview_aligns)

                jaccard_val = (
                    0.5
                    * (len(refids) + len(predids))
                    / max(
                        (
                            len(reference_comments)
                            + len(gen_comments)
                            - 0.5 * (len(refids) + len(predids))
                        ),
                        1,
                    )
                )
                metrics_for_reviews[k].append(
                    {
                        "recall": len(refids) / max(len(reference_comments), 1),
                        "precision": len(predids) / max(len(gen_comments), 1),
                        "pseudo_jaccard": jaccard_val,
                        "n_real": len(reference_comments),
                    }
                )

        # Aggregate all the metrics/alignments
        aligns_out = {
            k: list(itertools.chain(*aligns_for_reviews[k])) for k in generated_comments
        }
        all_aligns.append(aligns_for_reviews["all"])

        metrics = {
            k: {
                "n_reference": sum(x["n_real"] for x in metrics_for_reviews[k]),
                "n_generated": len(gen_comments),
                "n_total_aligns": len(aligns_out[k]),
                "recall": np.mean([x["recall"] for x in metrics_for_reviews[k]]),
                "precision": np.mean([x["precision"] for x in metrics_for_reviews[k]]),
                "pseudo_jaccard": np.mean(
                    [x["pseudo_jaccard"] for x in metrics_for_reviews[k]]
                ),
                "specificity_more": np.mean(
                    [x["relative_specificity"] == "more" for x in aligns_out[k]]
                ).tolist(),
                "specificity_less": np.mean(
                    [x["relative_specificity"] == "less" for x in aligns_out[k]]
                ).tolist(),
                "specificity_same": np.mean(
                    [x["relative_specificity"] == "same" for x in aligns_out[k]]
                ).tolist(),
            }
            for k, gen_comments in generated_comments.items()
        }

        all_metrics.append(metrics)
        logger.info("Metrics:\n{}".format(json.dumps(metrics)))

        all_results.append(
            {
                "doc_id": doc_id,
                "method": config["model_type"],
                "reference_comments": ref_comment_sets,
                "generated_comments": generated_comments,
                "alignments": aligns_out,
                "metrics": metrics,
                "metrics_for_reviews": metrics_for_reviews,
            }
        )

    # TODO: Kind of hacky getting consistent keys; should restructure so they're defined at top level
    subreview_types = set(itertools.chain(*[x.keys() for x in all_metrics]))

    for sr_type in subreview_types:
        logger.info("--- RESULTS FOR sr_type={sr_type} ---".format(sr_type=sr_type))
        metric_lists = {
            "pseudo_jaccard": [x[sr_type]["pseudo_jaccard"] for x in all_metrics],
            "recall": [
                x[sr_type]["recall"]
                for x in all_metrics
                if x[sr_type]["n_reference"] != 0
            ],
            "precision": [
                x[sr_type]["precision"]
                for x in all_metrics
                if x[sr_type]["n_reference"] != 0
            ],
            "n_generated": [x[sr_type]["n_generated"] for x in all_metrics],
        }
        logger.info(
            "Average (pseudo-)jaccard: {} ({:0.4f})".format(
                np.mean(metric_lists["pseudo_jaccard"]),
                np.std(metric_lists["pseudo_jaccard"]),
            )
        )
        logger.info(
            "Average recall: {} ({:0.4f})".format(
                np.mean(metric_lists["recall"]), np.std(metric_lists["recall"])
            )
        )
        logger.info(
            "Average precision: {} ({:0.4f})".format(
                np.mean(metric_lists["precision"]), np.std(metric_lists["precision"])
            )
        )
        logger.info(
            "Average n_generated: {} ({:0.4f})".format(
                np.mean(metric_lists["n_generated"]),
                np.std(metric_lists["n_generated"]),
            )
        )

    with open(os.path.join(config["output_dir"], "all_results.json"), "w") as f:
        json.dump(all_results, f, indent=4)
    with open(os.path.join(config["output_dir"], "all_aligns.json"), "w") as f:
        json.dump(all_aligns, f, indent=4)


if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        config = json.load(f)

    os.makedirs(config["output_dir"], exist_ok=True)

    init_logging(
        logfile=os.path.join(config["output_dir"], "logging_output.log"),
        level=logging.INFO,
    )

    main(config)
