import datetime
import hashlib
import json
import logging
import os
import sqlite3
import time
import warnings

import openai
import tiktoken
import tqdm

logger = logging.getLogger(__name__)


class Gpt3CacheClient:
    def __init__(self, cache_db_path):
        self.cache_db = self._init_cache_db(cache_db_path)

        if openai.api_key is None:
            if "OPENAI_API_KEY" not in os.environ:
                logger.error("Need OpenAI key in OPENAI_API_KEY")
            openai.api_key = os.environ["OPENAI_API_KEY"]

        self.tokenizer = None
        self.tokenizers_by_model = dict()

    def estimate_num_tokens(self, text, model="text-davinci-003"):
        return len(self._get_tokenizer(model).encode(text))

    def estimate_messages_num_tokens(self, messages, model="gpt-4-0613"):
        """Tries to estimate the number of tokens in a list of messages for the Chat Completions API."""
        # Note: useful information about overhead counts is available at:
        # https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
        # This function is partially adapted from that link, but might not stay
        # up-to-date with all model changes
        tokenizer = self._get_tokenizer(model)

        tokens_per_message = 0
        tokens_per_name = 0

        keymodel = model
        if keymodel.startswith("ft:"):
            keymodel = keymodel.split(":")[1]

        if keymodel in {
            "gpt-3.5-turbo-0613",
            "gpt-3.5-turbo-16k-0613",
            "gpt-4-0314",
            "gpt-4-32k-0314",
            "gpt-4-0613",
            "gpt-4-32k-0613",
        }:
            tokens_per_message = 3
            tokens_per_name = 1
        elif keymodel == "gpt-3.5-turbo-0301":
            tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
            tokens_per_name = -1  # if there's a name, the role is omitted
        elif "gpt-3.5-turbo" in keymodel:
            warnings.warn(
                f"gpt-3.5-turbo may update over time; use a more specific name if possible. Returning num tokens assuming gpt-3.5-turbo-0613.",
                UserWarning,
            )
            return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
        elif "gpt-4" in keymodel:
            warnings.warn(f"gpt-4 may update over time; use a more specific name if possible. Returning num tokens assuming gpt-4-0613.", UserWarning)
            return num_tokens_from_messages(messages, model="gpt-4-0613")
        else:
            raise ValueError(f"Unknown model {model}")

        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(tokenizer.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens

    def _get_tokenizer(self, model):
        if model not in self.tokenizers_by_model:
            keymodel = model
            if keymodel.startswith("ft:"):
                keymodel = keymodel.split(":")[1]
            self.tokenizers_by_model[model] = tiktoken.encoding_for_model(keymodel)
        return self.tokenizers_by_model[model]

    def __enter__(self):
        self.cache_db.__enter__()
        return self

    def __exit__(self, *args, **kwargs):
        self.cache_db.__exit__(*args, **kwargs)

    def close(self):
        self.cache_db.close()

    def _init_cache_db(self, cache_db_path):
        db = sqlite3.connect(cache_db_path)
        try:
            cur = db.cursor()
            cur.execute(
                """create table if not exists gpt3_cache (
                    model text not null,
                    prompt text not null,
                    temperature real not null,
                    top_p real not null,
                    max_tokens integer not null,
                    total_tokens integer not null,
                    frequency_penalty real not null,
                    presence_penalty real not null,
                    logprobs integer not null,
                    response_json text not null,
                    response_timestamp real
                    )"""
            )
            cur.execute("create index if not exists prompt_index on gpt3_cache (prompt)")

            cur.execute(
                """create table if not exists chat_gpt3_cache (
                    model text not null,
                    messages_json text not null,
                    temperature real not null,
                    top_p real not null,
                    max_tokens integer not null,
                    total_tokens integer not null,
                    frequency_penalty real not null,
                    presence_penalty real not null,
                    response_json text not null,
                    response_timestamp real
                    )"""
            )
            cur.execute("create index if not exists messages_json_index on chat_gpt3_cache (messages_json)")

            cur.execute(
                """create table if not exists embedding_gpt3_cache (
                    model text not null,
                    text text not null,
                    total_tokens integer not null,
                    response_json text not null,
                    response_timestamp real
                    )"""
            )
            cur.execute("create index if not exists embedding_input_text_index on embedding_gpt3_cache (text)")
            db.commit()
            return db
        except Exception as e:
            db.close()
            raise e

    def get_gpt3_result(self, *args, **kwargs):
        """Deprecated. Use prompt_completion() instead."""
        return self.prompt_completion(*args, **kwargs)

    def prompt_completion(
        self,
        model,
        prompt,
        temperature,
        max_tokens,
        top_p,
        frequency_penalty,
        presence_penalty,
        prompt_token_count=-1,
        logprobs=0,
    ):
        """Works like openai.Completion.create, but adds a caching layer."""
        if prompt_token_count < 0:
            prompt_token_count = self.estimate_num_tokens(prompt, model)

        db_keyvals = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "logprobs": logprobs,
        }
        cur = self.cache_db.cursor()

        cache_json = None
        from_cache = False
        # Cache only makes sense if temperature==0 (deterministic result)
        if temperature == 0.0:
            select_keyvals = db_keyvals.copy()
            select_keyvals["prompt_token_count"] = prompt_token_count
            dbrecs = cur.execute(
                """select response_json from gpt3_cache
                where
                model = :model and
                prompt = :prompt and
                temperature = :temperature and
                ((:prompt_token_count+max_tokens) > total_tokens or max_tokens = :max_tokens) and
                total_tokens <= (:prompt_token_count+:max_tokens) and
                top_p = :top_p and
                frequency_penalty = :frequency_penalty and
                presence_penalty = :presence_penalty and
                logprobs >= :logprobs""",
                select_keyvals,
            ).fetchall()
            if len(dbrecs) == 1:
                cache_json = dbrecs[0][0]
            elif len(dbrecs) >= 2:
                logger.warning("Got {} recs for gpt3 query when only one was expected.".format(len(dbrecs)))
                cache_json = dbrecs[0][0]
        if cache_json is None:
            logger.debug("UNCACHED prompt completion")
            resp = openai.Completion.create(**db_keyvals)
            insert_keyvals = db_keyvals.copy()
            cache_json = json.dumps(resp)
            insert_keyvals["response_json"] = cache_json
            insert_keyvals["response_timestamp"] = datetime.datetime.timestamp(datetime.datetime.utcnow())
            insert_keyvals["total_tokens"] = resp["usage"]["total_tokens"]
            cur.execute(
                """INSERT INTO gpt3_cache ( model,  prompt,  temperature,  top_p,  max_tokens,  frequency_penalty,  presence_penalty,  logprobs,  response_json,  response_timestamp,  total_tokens)
                   VALUES                 (:model, :prompt, :temperature, :top_p, :max_tokens, :frequency_penalty, :presence_penalty, :logprobs, :response_json, :response_timestamp, :total_tokens)""",
                insert_keyvals,
            )
            self.cache_db.commit()
        else:
            from_cache = True

        resp = json.loads(cache_json)
        if from_cache:
            resp["usage"]["uncached_total_tokens"] = 0
        else:
            resp["usage"]["uncached_total_tokens"] = resp["usage"]["total_tokens"]
        return resp

    def chat_completion(
        self,
        model,
        messages,
        temperature,
        max_tokens,
        top_p,
        frequency_penalty,
        presence_penalty,
        messages_token_count=-1,
        max_retries=3,
    ):
        """Works like openai.ChatCompletion.create, but adds a caching layer."""

        # Sort keys when serializing to maximize cache hits
        messages_json = json.dumps(messages, sort_keys=True)

        if messages_token_count < 0:
            # messages_token_count = sum(self.estimate_num_tokens(x["content"], model) for x in messages)
            messages_token_count = self.estimate_messages_num_tokens(messages)

        if max_tokens <= 0:
            raise MaxTokensBoundError("max_tokens must be > 0")

        db_keyvals = {
            "model": model,
            "messages_json": messages_json,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
        }
        cur = self.cache_db.cursor()

        cache_json = None
        from_cache = False
        # Cache only makes sense if temperature==0 (deterministic result)
        if temperature == 0.0:
            select_keyvals = db_keyvals.copy()
            select_keyvals["messages_token_count"] = messages_token_count

            dbrecs = cur.execute(
                """select response_json from chat_gpt3_cache
                where
                model = :model and
                messages_json = :messages_json and
                temperature = :temperature and
                ((:messages_token_count+max_tokens) > total_tokens or max_tokens = :max_tokens) and
                total_tokens <= (:messages_token_count+:max_tokens) and
                top_p = :top_p and
                frequency_penalty = :frequency_penalty and
                presence_penalty = :presence_penalty
                """,
                select_keyvals,
            ).fetchall()

            if len(dbrecs) == 1:
                cache_json = dbrecs[0][0]
            elif len(dbrecs) >= 2:
                logger.warning("Got {} recs for gpt3 query when only one was expected.".format(len(dbrecs)))
                cache_json = dbrecs[0][0]
        if cache_json is None:
            logger.debug("UNCACHED chat completion")

            model_keyvals = db_keyvals.copy()
            del model_keyvals["messages_json"]
            model_keyvals["messages"] = messages

            resp = None
            while resp is None and max_retries >= 0:
                max_retries -= 1
                try:
                    resp = openai.ChatCompletion.create(**model_keyvals)
                except openai.error.RateLimitError:
                    logger.warning("Rate limit error on openai request, waiting 60 seconds and trying again")
                    time.sleep(60)
                except openai.error.APIError as e:
                    if e.json_body["error"]["code"] == 502:
                        logger.error("Bad gateway on openai request, waiting 10 seconds and trying again")
                        time.sleep(10)
                    else:
                        raise e
                except openai.error.Timeout:
                    logger.error("Timeout on openai request, retrying immediately")

            insert_keyvals = db_keyvals.copy()
            cache_json = json.dumps(resp)
            insert_keyvals["response_json"] = cache_json
            insert_keyvals["response_timestamp"] = datetime.datetime.timestamp(datetime.datetime.utcnow())
            insert_keyvals["total_tokens"] = resp["usage"]["total_tokens"]
            cur.execute(
                """INSERT INTO chat_gpt3_cache ( model,  messages_json,  temperature,  top_p,  max_tokens,  frequency_penalty,  presence_penalty,  response_json,  response_timestamp,  total_tokens)
                   VALUES                      (:model, :messages_json, :temperature, :top_p, :max_tokens, :frequency_penalty, :presence_penalty, :response_json, :response_timestamp, :total_tokens)""",
                insert_keyvals,
            )
            self.cache_db.commit()
        else:
            from_cache = True

        resp = json.loads(cache_json)
        if from_cache:
            resp["usage"]["uncached_total_tokens"] = 0
        else:
            resp["usage"]["uncached_total_tokens"] = resp["usage"]["total_tokens"]
        return resp

    def embeddings(
        self,
        model,
        texts,
        max_retries=3,
    ):
        if len(texts) == 0:
            return None
        resps = [self.one_embedding(model, x, max_retries=max_retries) for x in texts]
        full_resp = resps[0].copy()
        full_resp["data"] = []
        for idx, resp in enumerate(resps):
            resp["data"][0]["index"] = idx
            full_resp["data"].append(resp["data"][0])
        full_resp["usage"]["prompt_tokens"] = sum([x["usage"]["prompt_tokens"] for x in resps])
        full_resp["usage"]["total_tokens"] = sum([x["usage"]["total_tokens"] for x in resps])
        return full_resp

    def one_embedding(
        self,
        model,
        text,
        max_retries=3,
    ):
        """Works like openai.Embedding.create, but adds a caching layer."""

        db_keyvals = {
            "model": model,
            "text": text,
        }
        cur = self.cache_db.cursor()

        cache_json = None
        from_cache = False

        select_keyvals = db_keyvals.copy()

        dbrecs = cur.execute(
            """select response_json from embedding_gpt3_cache
            where
            model = :model and
            text = :text
            """,
            select_keyvals,
        ).fetchall()

        if len(dbrecs) == 1:
            cache_json = dbrecs[0][0]
        elif len(dbrecs) >= 2:
            logger.warning("Got {} recs for gpt3 query when only one was expected.".format(len(dbrecs)))
            cache_json = dbrecs[0][0]

        if cache_json is None:
            logger.debug("UNCACHED embedding")

            model_keyvals = db_keyvals.copy()

            resp = None
            while resp is None and max_retries >= 0:
                try:
                    resp = openai.Embedding.create(input=[model_keyvals["text"]], model=model_keyvals["model"])
                except openai.error.RateLimitError:
                    logger.warning("Rate limit error on openai request, waiting 60 seconds and trying again")
                    time.sleep(60)
                    max_retries -= 1

            insert_keyvals = db_keyvals.copy()
            cache_json = json.dumps(resp)
            insert_keyvals["response_json"] = cache_json
            insert_keyvals["response_timestamp"] = datetime.datetime.timestamp(datetime.datetime.utcnow())
            insert_keyvals["total_tokens"] = resp["usage"]["total_tokens"]
            cur.execute(
                """INSERT INTO embedding_gpt3_cache ( model,  text,  response_json,  response_timestamp,  total_tokens)
                   VALUES                           (:model, :text, :response_json, :response_timestamp, :total_tokens)""",
                insert_keyvals,
            )
            self.cache_db.commit()
        else:
            from_cache = True

        resp = json.loads(cache_json)
        if from_cache:
            resp["usage"]["uncached_total_tokens"] = 0
        else:
            resp["usage"]["uncached_total_tokens"] = resp["usage"]["total_tokens"]
        return resp


class GptLogicError(Exception):
    pass


class MaxTokensBoundError(ValueError):
    pass
