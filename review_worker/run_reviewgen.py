import os
import base64
import traceback
import itertools
import datetime
import time
import argparse
import json
import sqlite3
import shutil
import sys
import random
import tempfile
import subprocess
import logging

from grobid_client.grobid_client import GrobidClient

import tiktoken

import boto3
from botocore.exceptions import BotoCoreError, ClientError

import aries.util.gpt3
import aries.util.s2orc
from aries.alignment.doc_edits import DocEdits
from aries.review_generation.multi_agent import MultiAgentGroup, GptChatBot, make_chunked_paper_diff
from aries.util.color import colorify, colorprint
from aries.util.logging import init_logging

logger = logging.getLogger(__name__)

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = "/app_data"
URL_BASE = 'http://localhost:8080'
DATABASE = f"{DATA_DIR}/app.db"
ERROR_EMAIL = None
INFO_EMAIL = None
OUTGOING_EMAIL = None


def reviewgen_v21_bare_parallel_chunked(config, paper_chunks, taxonomy_element, prompts):
    rts = []
    for chunk in paper_chunks:
        bot = GptChatBot(
            config["gpt3_cache_db_path"],
            model=config["gpt_model"],
            system_prompt=prompts["system_prompt_v1"],
            max_tokens=config["gpt_default_max_length"],
        )
        bot.paper_chunk = chunk
        rt, resp = bot.chat(prompts["task_prompt_set1_v1"].format(source_paper_chunk=bot.paper_chunk))
        if prompts.get("task_prompt_set1_v2", None) is not None:
            rt, resp = bot.chat(prompts["task_prompt_set1_v2"].format(comment_type=taxonomy_element))
        rts.append(rt)
        # print(rt)
    bot = GptChatBot(
        config["gpt3_cache_db_path"],
        model=config["gpt_model"],
        system_prompt=prompts["combiner_system_prompt"],
        max_tokens=config["gpt_default_max_length"],
    )
    merged_review, resp = bot.chat(prompts["combiner_task_prompt"].format(reviewlists="\n-----\n".join(rts)))
    merged_review, resp = bot.chat(prompts["combiner_task_prompt_v2"])
    logger.info(merged_review)
    return merged_review

def reviewgen_v22_parallel_chunked(config, paper_chunks, taxonomy_element, prompts):
    rts = []
    for chunk in paper_chunks:
        bot = GptChatBot(
            config["gpt3_cache_db_path"],
            model=config["gpt_model"],
            system_prompt=prompts["system_prompt_v1"],
            max_tokens=config["gpt_default_max_length"],
        )
        bot.paper_chunk = chunk
        bot.messages.append({"role": "user", "content": prompts["task_prompt_set1_v1"].format(source_paper_chunk=bot.paper_chunk)})
        bot.messages.append({"role": "assistant", "content": "Ready"})
        bot.chat(prompts["task_prompt_set1_v2"].format(comment_type=taxonomy_element))
        rt, resp = bot.chat(prompts["task_prompt_set1_v3"])
        rts.append(rt)
        # print(rt)
    bot = GptChatBot(
        config["gpt3_cache_db_path"],
        model=config["gpt_model"],
        system_prompt=prompts["combiner_system_prompt"],
        max_tokens=config["gpt_default_max_length"],
    )
    merged_review, resp = bot.chat(prompts["combiner_task_prompt"].format(reviewlists="\n-----\n".join(rts)))
    cur_review = merged_review
    if prompts.get("task_prompt_set2_v1", None) is not None:
        for chunk in paper_chunks:
            bot = GptChatBot(
                config["gpt3_cache_db_path"],
                model=config["gpt_model"],
                system_prompt=prompts["system_prompt_v2"],
                max_tokens=config["gpt_default_max_length"],
            )
            bot.paper_chunk = chunk
            bot.messages.append({"role": "user", "content": prompts["task_prompt_set2_v1"].format(source_paper_chunk=bot.paper_chunk)})
            bot.messages.append({"role": "assistant", "content": "Ready"})
            bot.chat(prompts["task_prompt_set2_v2"].format(comment_type=taxonomy_element, review_comments=cur_review))
            rt, resp = bot.chat(prompts["task_prompt_set2_v3"])
            # if chunk == paper_chunks[-1]:
            #    rt, resp = bot.chat("Identify one particular comment that is the most important for improving the paper.  Carefully consider the paper's goals and claims and pick the comment that represents the greatest weakness of the paper.")
            cur_review = rt
            logger.info(rt)
    else:
        logger.warning("No refinement prompt was given; skipping refinement stage")
        cur_review, _ = bot.chat(
            prompts.get(
                "task_prompt_set2_v3", "Write the final list of review comments as a JSON list of strings.  Do not include any additional commentary."
            )
        )
    return cur_review


def reviewgen_v24_multi_agent(config, paper_chunks, taxonomy_element, prompts):

    extra_agents = config["experts"]

    swarm = MultiAgentGroup(
        config,
        None,
        config["gpt_model"],
        paper_chunk_size=config["paper_chunk_size"],
        prompts=prompts,
        max_tokens=config["gpt_default_max_length"],
        quiet=False,
        doc_mods=None,
        use_history_pruning=True,
        taxonomy="",
        master_chunk_type=config["master_chunk_type"],
        extra_bots=extra_agents,
        raw_paper_chunks=paper_chunks,
        # color_format='html',
    )
    expert_name_substitutions = {x["name"]: swarm.extra_bot_experts[eidx].agent_name for eidx, x in enumerate(config["experts"])}
    master_task_prompt = prompts["task_prompt_set1_v1"].format(comment_type=taxonomy_element, **expert_name_substitutions)
    rt, resp = swarm.ask_swarm_question(master_task_prompt, pre_prompt="")
    tmpbot = GptChatBot(
        config["gpt3_cache_db_path"],
        model="gpt-4-0613",
        system_prompt='Instructions:\nYou will be given an output from a review-generating AI.  Your job is to determine whether the output contains a finalized list of review comments.  If so, output the review comments verbatim in a JSON list of strings.  If not, write "No comments."  Note that if the list appears incomplete--e.g., it starts with a number other than 1, etc--you should write "No comments." anyway.',
        max_tokens=2048,
    )
    rt, resp = tmpbot.chat("Output:\n" + rt)
    if rt.strip().strip('"') == "No comments.":
        rt, resp = swarm.ask_swarm_question(prompts["task_prompt_set1_v2"], pre_prompt="")
        try:
            _ = json.loads(rt)
        except json.decoder.JSONDecodeError:
            rt, resp = swarm.ask_swarm_question(prompts["task_prompt_set1_v2"] + "\n\nMake sure to use the specified JSON format.", pre_prompt="")
    else:
        print(rt)
    try:
        review1 = rt[rt.index("[") :].strip().strip("`")
    except Exception as e:
        logger.exception("FAILED TO PARSE MODEL JSON OUTPUT")
        review1 = "null"

    if config.get("skip_refinement"):
        return review1

    final_comments = []

    for comment in json.loads(review1):
        swarm = MultiAgentGroup(
            config,
            None,
            config["gpt_model"],
            paper_chunk_size=config["paper_chunk_size"],
            prompts=prompts,
            max_tokens=config["gpt_default_max_length"],
            quiet=False,
            doc_mods=None,
            use_history_pruning=True,
            master_chunk_type=config["master_chunk_type"],
            taxonomy="",
            raw_paper_chunks=paper_chunks,
        )
        rt, resp = swarm.ask_swarm_question(
            prompts["task_prompt_set2_v1"].format(comment_type=taxonomy_element, review_comments=comment),
            pre_prompt="",
        )
        rt, resp = swarm.ask_swarm_question(prompts["task_prompt_set2_v2"], pre_prompt="")
        try:
            comment = json.loads(rt)
        except json.decoder.JSONDecodeError:
            logger.exception("Bad comment JSON:")
            comment = None

        if comment is not None:
            if "revised_comments" in comment:
                assert "revised_comment" not in comment
                for c in comment["revised_comments"]:
                    final_comments.append({"revised_comment": c})
            else:
                final_comments.append(comment)
    return json.dumps([x["revised_comment"] for x in final_comments if x["revised_comment"] is not None])

def reviewgen_v25_generic_multi_agent(config, doc_id, paper_chunks, taxonomy_element, prompts):
    extra_agents = config["experts"]

    swarm = MultiAgentGroup(
        config,
        None,
        config["gpt_model"],
        paper_chunk_size=config["paper_chunk_size"],
        prompts=prompts,
        max_tokens=config["gpt_default_max_length"],
        quiet=False,
        doc_mods=None,
        use_history_pruning=False,
        taxonomy="",
        master_chunk_type=config["master_chunk_type"],
        extra_bots=extra_agents,
        raw_paper_chunks=paper_chunks,
        # color_format='html',
    )
    expert_name_substitutions = {x["name"]: swarm.extra_bot_experts[eidx].agent_name for eidx, x in enumerate(config["experts"])}
    master_task_prompt = prompts["task_prompt_set1_v1"].format(comment_type=taxonomy_element, **expert_name_substitutions)
    rt, resp = swarm.ask_swarm_question(master_task_prompt, pre_prompt="")
    tmpbot = GptChatBot(
        config["gpt3_cache_db_path"],
        model="gpt-4-0613",
        system_prompt='Instructions:\nYou will be given an output from a review-generating AI.  Your job is to determine whether the output contains a finalized list of review comments.  If so, output the review comments verbatim in a JSON list of strings.  If not, write "No comments."  Note that if the list appears incomplete--e.g., it starts with a number other than 1, etc--you should write "No comments." anyway.',
        max_tokens=2048,
    )
    rt, resp = tmpbot.chat("Output:\n" + rt)
    if rt.strip().strip('"') == "No comments.":
        rt, resp = swarm.ask_swarm_question(prompts["task_prompt_set1_v2"], pre_prompt="")
        try:
            _ = json.loads(rt)
        except json.decoder.JSONDecodeError:
            rt, resp = swarm.ask_swarm_question(prompts["task_prompt_set1_v2"] + "\n\nMake sure to use the specified JSON format.", pre_prompt="")
    else:
        print(rt)
    try:
        review1 = rt[rt.index("[") :].strip().strip("`")
    except Exception as e:
        logger.exception("FAILED TO PARSE MODEL JSON OUTPUT")
        review1 = "[]"

    final_comments = []

    swarm = MultiAgentGroup(
        config,
        None,
        config["gpt_model"],
        paper_chunk_size=config["paper_chunk_size"],
        prompts=prompts,
        max_tokens=config["gpt_default_max_length"],
        quiet=False,
        doc_mods=None,
        use_history_pruning=True,
        master_chunk_type=config["master_chunk_type"],
        raw_paper_chunks=paper_chunks,
        taxonomy="",
    )
    rt, resp = swarm.ask_swarm_question(
        prompts["task_prompt_set2_v1"].format(comment_type=taxonomy_element, review_comments=review1),
        pre_prompt="",
    )
    rt, resp = swarm.ask_swarm_question(prompts["task_prompt_set2_v2"], pre_prompt="")
    final_comments = json.loads(rt)
    return json.dumps(final_comments)


def reviewgen_v26_specialized_multi_agent(config, paper_chunks, taxonomy_element, prompt_sets, doc_id=""):
    comments_by_type = dict()
    default_prompts = config.get("default_prompts", dict())
    for pset in prompt_sets:
        pset = pset.copy()
        for k, v in default_prompts.items():
            if k not in pset:
                pset[k] = v

        new_config = config.copy()
        new_config["experts"] = pset.get("experts", [])

        max_retries = 4

        for retry in range(max_retries):
            try:
                rev = reviewgen_v24_multi_agent(new_config, paper_chunks, taxonomy_element, pset)
                break
            except aries.util.gpt3.MaxTokensBoundError:
                logger.exception("Input token limit reached for {}.".format(doc_id))
                rev = "[]"
                break
            except Exception as e:
                logger.exception("Failed to make review for {}.".format(doc_id))
                rev = "[]"
                send_email(
                    ERROR_EMAIL,
                    f"A review failed",
                    (
                        f"A sub-review couldn't be generated for {doc_id}, name={pset.get('name', 'unknown')}.  This is attempt {retry+1} of {max_retries}.  It failed with the following error:\n\n{e}\n\n{traceback.format_exc()}"
                    ).replace("\n", "<br>"),
                )
                continue
        comments_by_type[pset["name"]] = json.loads(rev)

    comments_by_type['all'] = list(itertools.chain(*comments_by_type.values()))
    final_rev = json.dumps(comments_by_type)
    return final_rev


def get_s2orc_for_pdf(pdf_path):
    pdf_base = os.path.basename(pdf_path)[: -len(".pdf")]
    if os.path.exists(f"{DATA_DIR}/s2orc/{pdf_base}.json"):
        with aries.util.s2orc.S2orcFetcherFilesystem(f"{DATA_DIR}/s2orc") as fetcher:
            return aries.util.s2orc.load_s2orc(pdf_base, fetcher)

    s2orc_script_path = os.path.join(BASE_PATH, "s2orc-doc2json/doc2json/grobid2json/process_pdf.py")
    # Run grobid on the pdf_path
    with tempfile.TemporaryDirectory(prefix="tmp", dir=DATA_DIR) as tmpdir:
        pdf_tmp_path = os.path.join(tmpdir, "paper.pdf")
        shutil.copyfile(pdf_path, pdf_tmp_path)

        print(os.listdir(tmpdir))

        retcode = subprocess.call(
            [
                "python",
                s2orc_script_path,
                "-i",
                pdf_tmp_path,
                "-t",
                tmpdir,
                "-o",
                tmpdir,
                "--grobid_config",
                os.path.join(BASE_PATH, "grobid_client_config.json"),
            ]
        )
        shutil.copyfile(os.path.join(tmpdir, f"paper.json"), f"{DATA_DIR}/s2orc/{pdf_base}.json")
        with aries.util.s2orc.S2orcFetcherFilesystem(f"{DATA_DIR}/s2orc") as fetcher:
            return aries.util.s2orc.load_s2orc(pdf_base, fetcher)
    return None


def make_liang_etal_paper_blob(s2orc1, model="gpt-4"):
    # Adapted from https://github.com/Weixin-Liang/LLM-scientific-feedback/blob/main/main.py
    title = s2orc1["title"]
    abstract = s2orc1["abstract"]
    fig_table_captions = "\n".join([x["text"].strip() for x in s2orc1["pdf_parse"]["ref_entries"].values() if x["type_str"] in ("table", "figure")])
    main_content = ""

    def format_source_text(para_text, section_name="unknown"):
        s = ""
        if section_name != "unknown":
            s += "\nSection title: {}\n".format(section_name)
        # s += "\nparagraph id: {}".format(para_id)
        s += para_text
        s += "\n\n"
        return s

    for para_idx, para_obj in enumerate(s2orc1["pdf_parse"]["body_text"]):
        if para_obj["text"].strip() == "":
            continue

        section_name = (para_obj["section"] or "unknown").strip()

        main_content += format_source_text(para_obj["text"].strip(), section_name=section_name)

    paper_blob = f"""Abstract:
```
{abstract}
```

Figures/Tables Captions:
```
{fig_table_captions}
```

Main Content:
```
{main_content}
```"""

    max_tokens = 6500
    tokenizer = tiktoken.encoding_for_model(model)

    paper_blob = tokenizer.decode(tokenizer.encode(paper_blob)[:max_tokens])

    # Add back the closing ``` if it was truncated
    if not paper_blob.endswith("```"):
        paper_blob += "\n```"
    return paper_blob


def reviewgen_v27_liang_etal(config, s2orc1, prompts):
    truncated_paper = make_liang_etal_paper_blob(s2orc1, model=config["gpt_model"])
    text_to_send = f"""Your task now is to draft a high-quality review outline for a top-tier Machine Learning (ML) conference for a submission titled "{s2orc1['title']}":

{truncated_paper}


======
Your task:
Compose a high-quality peer review of an ML paper submitted to a top-tier ML conference on OpenReview.

Start by "Review outline:".
And then:
"1. Significance and novelty"
"2. Potential reasons for acceptance"
"3. Potential reasons for rejection", List 4 key reasons. For each of 4 key reasons, use **>=2 sub bullet points** to further clarify and support your arguments in painstaking details.
"4. Suggestions for improvement", List 4 key suggestions.

Be thoughtful and constructive. Write Outlines only.

"""
    bot = GptChatBot(
        config["gpt3_cache_db_path"],
        model=config["gpt_model"],
        system_prompt="You are ChatGPT, a large language model trained by OpenAI. Answer as concisely as possible.",
        max_tokens=config["gpt_default_max_length"],
    )
    review_outline, resp = bot.chat(text_to_send)
    logger.info("REVIEW OUTLINE: {}".format(review_outline))

    bot = GptChatBot(
        config["gpt3_cache_db_path"],
        model=config["gpt_model"],
        system_prompt="You are ChatGPT, a large language model trained by OpenAI. Answer the user's requests using well-formed JSON, and copy the input verbatim unless otherwise specified.",
        max_tokens=config["gpt_default_max_length"],
    )
    json_review, resp = bot.chat(
        """Your goal is to identify the key concerns raised in the review, focusing only on potential
reasons for rejection.

Please provide your analysis in JSON format, including a concise summary, and the exact
wording from the review.

Submission Title: {title}

=====Review:
```
{review}
```

=====
Example JSON format:
{{
    "1": {{"summary": "<your concise summary>", "verbatim": "<concise, copy the exact wording in the review>"}},
    "2": ...
}}

Analyze the review and provide the key concerns in the format specified above. Ignore minor
issues like typos and clarifications. Output only json.""".format(
            title=s2orc1["title"], review=review_outline
        )
    )
    json_review = json.dumps([x["verbatim"] for x in json.loads(json_review).values()], indent=4)
    logger.info(json_review)
    return json_review


def get_paper_chunks(pdf_path, config):
    doc_edits = None

    s2orc1 = get_s2orc_for_pdf(pdf_path)
    s2orc2 = s2orc1
    doc_edits = DocEdits.from_list(
        s2orc1, s2orc2, [{"source_idxs": [x], "target_idxs": [x], "edit_id": x} for x in range(len(s2orc1["pdf_parse"]["body_text"]))]
    )

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

    return paper_chunks, s2orc1


def make_reviews(pdf_hash):
    pdf_path = os.path.join(DATA_DIR, "uploads", pdf_hash + ".pdf")

    with open(os.path.join(BASE_PATH, "review_prompts.json"), "r") as f:
        config = json.load(f)

    paper_chunks = None
    cur_chunk_size = -1

    is_success = True

    with sqlite3.connect(DATABASE) as db:
        cur = db.cursor()
        for method, subconfig in config["configs"].items():
            subconfig["gpt3_cache_db_path"] = config["gpt3_cache_db_path"]
            cur.execute("SELECT * FROM reviews WHERE pdf_hash=? AND method=?", (pdf_hash, method))
            row = cur.fetchone()
            if row is None:
                if paper_chunks is None or cur_chunk_size != subconfig["paper_chunk_size"]:
                    paper_chunks, s2orc1 = get_paper_chunks(pdf_path, subconfig)
                    cur_chunk_size = subconfig["paper_chunk_size"]
                review = make_review(subconfig, paper_chunks, s2orc1, doc_id=pdf_hash)
                if isinstance(review, dict):
                    review = list(itertools.chain(*review.values()))
                if review is not None and len(review) != 0:
                    cur.execute(
                        'INSERT INTO reviews (pdf_hash, method, review_json, completed_time) VALUES (?, ?, ?, datetime("now"))',
                        (pdf_hash, method, json.dumps(review)),
                    )
                    db.commit()
                else:
                    logger.warning(f"Review for {pdf_hash} with method {method} was empty; possible error.")
                    is_success = False

    return is_success


def make_review(config, paper_chunks, s2orc1, doc_id=""):
    try:
        if config["model_type"] == "gpt_bare_independent_chunk":
            rev1 = reviewgen_v21_bare_parallel_chunked(config, paper_chunks, "", config["prompts"])
        elif config["model_type"] == "gpt_independent_chunk":
            rev1 = reviewgen_v22_parallel_chunked(config, paper_chunks, "", config["prompts"])
        elif config["model_type"] == "gpt_specialized_multi_agent":
            rev1 = reviewgen_v26_specialized_multi_agent(config, paper_chunks, "", config["prompt_sets"], doc_id=doc_id)
        elif config["model_type"] == "gpt_generic_multi_agent":
            rev1 = reviewgen_v25_generic_multi_agent(config, doc_id, paper_chunks, "", config["prompts"])
        elif config["model_type"] == "gpt_liang_etal":
            rev1 = reviewgen_v27_liang_etal(config, s2orc1, config["prompts"])
        else:
            raise ValueError(f"Unknown model type {config['model_type']}")
    except Exception as e:
        logger.exception(f"Failed to generate review for {doc_id}")
        send_email(
            ERROR_EMAIL,
            f"A review failed",
            (f"A review couldn't be generated for {doc_id}.  It failed with the following error:\n\n{e}\n\n{traceback.format_exc()}").replace(
                "\n", "<br>"
            ),
        )
        return None

    try:
        rev1 = json.loads(rev1)
    except:
        rev1 = None
        logger.error(f"Failed to parse review: {rev1}")

    return rev1


def send_email(to, subject, body):
    if to is None or OUTGOING_EMAIL is None:
        logger.warning("Not sending email because no email address is configured")
        return

    # Try to send the email.
    try:
        client = boto3.client("ses", region_name="us-west-2")

        # Provide the contents of the email.
        response = client.send_email(
            Destination={
                "ToAddresses": [
                    to,
                ],
            },
            Message={
                "Body": {
                    "Html": {
                        "Charset": "UTF-8",
                        "Data": body,
                    },
                    "Text": {
                        "Charset": "UTF-8",
                        "Data": body,
                    },
                },
                "Subject": {
                    "Charset": "UTF-8",
                    "Data": subject,
                },
            },
            Source=OUTGOING_EMAIL,
        )

    except ClientError as e:
        logger.exception("Failed to send notification email to %s", to)
    except Exception as e:
        logger.exception("Failed with some other error to send notification email to %s", to)
    else:
        logger.info("Email sent to %s with message ID %s" % (to, response["MessageId"]))


def check_work_queue(is_cold_boot=False):
    # Check the queue for any new submissions
    with sqlite3.connect(DATABASE) as db:
        cur = db.cursor()
        cur.execute("select rowid, pdf_hash, status, worker_pid, submitted_time, notification_email from work_queue where status = ?", ("new",))
        key_idxs = {"rowid": 0, "pdf_hash": 1, "status": 2, "worker_pid": 3, "submitted_time": 4, "notification_email": 5}
        work_row = None
        rows = cur.fetchall()
        for row in rows:
            row_id, pdf_hash, status, worker_pid, submitted_time, notification_email = row
            work_row = row

            cur.execute(
                "update work_queue set status = ?, worker_pid = ? where pdf_hash = ? and status = ?", ("processing", os.getpid(), pdf_hash, status)
            )
            db.commit()

        if work_row is None:
            return

        notif_rows = []
        for row in rows:
            if row[key_idxs["pdf_hash"]] == work_row[key_idxs["pdf_hash"]] and row[key_idxs["status"]] == "new":
                notif_rows.append(row)

        row_id, pdf_hash, status, worker_pid, submitted_time, notification_email = work_row

        logger.info(f"Processing work queue item {pdf_hash}")
        send_email(
            INFO_EMAIL,
            f"A paper was submitted for review",
            f"A paper was submitted for review with hash {pdf_hash} and is being processed by worker {os.getpid()}.",
        )

        try:
            is_makereviews_success = make_reviews(pdf_hash)
            logger.info(f"Completed work queue item {pdf_hash}")

            should_do_notif = is_makereviews_success

            result_url = "{url_base}/result/{pdf_hash}".format(url_base=URL_BASE, pdf_hash=pdf_hash)
            workbot_debug_email = f'You can view your paper feedback at <a href="{result_url}">{result_url}</a>.'
            workbot_debug_email += f"<br><br>Debug info:"
            workbot_debug_email += f"<br>is_makereviews_success={is_makereviews_success}"

            send_email(INFO_EMAIL, f"A review is ready", workbot_debug_email.replace("\n", "<br>"))

            # Send notification email
            if should_do_notif:
                for row in notif_rows:
                    # Check to make sure the status hasn't changed (e.g., by some other replica)
                    cur.execute("select rowid, status from work_queue where rowid = ?", (row[key_idxs["rowid"]],))
                    tmprow = cur.fetchone()
                    if tmprow is None:
                        continue
                    if tmprow[1] != "processing":
                        logger.warning("Skipping notification email for rowid %d because status is %s", row[key_idxs["rowid"]], tmprow[1])
                        continue
                    cur.execute(
                        'update work_queue set status = ?, completed_time = datetime("now") where rowid = ?', ("completed", row[key_idxs["rowid"]])
                    )
                    db.commit()
                    notification_email = row[key_idxs["notification_email"]]
                    if notification_email is not None:
                        user_id = (
                            base64.b64encode(notification_email.encode("utf-8")).decode("utf-8").replace("+", "-").replace("/", "_").replace("=", "")
                        )
                        survey_url = "{url_base}/survey/{pdf_hash}/{user_id}".format(url_base=URL_BASE, pdf_hash=pdf_hash, user_id=user_id)
                        send_email(
                            notification_email,
                            f"Your review is ready",
                            f'You can view your paper feedback at <a href="{survey_url}">{survey_url}</a>.  Please take a few minutes to fill out the corresponding survey to help us learn what\'s good or bad about the generated feedback.',
                        )
        except:
            logger.error(f"Failed to process work queue item {pdf_hash}")
            cur.execute('update work_queue set status = ?, completed_time = datetime("now") where pdf_hash = ?', ("failed", pdf_hash))
            db.commit()
            raise


def init_db():
    with sqlite3.connect(DATABASE) as db:
        with open(f'{DATA_DIR}/schema.sql', mode='r') as f:
            db.cursor().executescript(f.read())
        db.commit()


if __name__ == "__main__":
    init_logging(
        # logfile=os.path.join(config["output_dir"], "logging_output.log"),
        level=logging.INFO,
    )

    logger.info("Initializing database...")
    init_db()

    logger.info("Beginning work loop")

    loopnum = 0
    while True:
        try:
            check_work_queue(is_cold_boot=(loopnum == 0))
            loopnum += 1
        except KeyboardInterrupt:
            logger.warning("Keyboard interrupt, exiting gracefully")
            break
        except Exception as e:
            logger.exception("Error in work queue processing")
            try:
                send_email(
                    ERROR_EMAIL,
                    f"The reviewer failed badly",
                    (f"The review system failed very badly.  It failed with the following error:\n\n{e}\n\n{traceback.format_exc()}").replace(
                        "\n", "<br>"
                    ),
                )
            except:
                logger.exception("Extra error sending email")
        time.sleep(10)
