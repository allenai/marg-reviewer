import json
import collections
import logging
import os
import sys

import openai
import tqdm

import aries.util.data
import aries.util.edit
import aries.util.gpt3
from aries.util.color import colorify, colorprint
from aries.util.data import index_by, iter_jsonl_files, openc

logger = logging.getLogger(__name__)


class GptChatBot:
    def __init__(self, cache_db_path, model="gpt-4-0613", system_prompt=None, max_tokens=1024):
        self.cache_db_path = cache_db_path
        self.model = model
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.quiet = False
        self.name = None
        self.messages = []

        self.reset()

    def reset(self):
        self.messages = []
        if self.system_prompt is not None:
            self.messages.append({"role": "system", "content": self.system_prompt})

    def chat(self, prompt, role="user", max_tokens=None, gptcli=None):
        if max_tokens is None:
            max_tokens = self.max_tokens
        self.messages.append({"role": role, "content": prompt})
        if self.name is not None:
            self.messages[-1]["name"] = self.name
        result_text, response = self.run_prompt_basic(self.model, self.messages, max_tokens, gptcli=gptcli)
        self.messages.append(response["choices"][0]["message"].copy())
        return result_text, response

    def run_prompt_basic(self, model, messages, max_tokens, gptcli=None):
        if gptcli is None:
            with aries.util.gpt3.Gpt3CacheClient(self.cache_db_path) as gptcli2:
                return self.run_prompt_basic(model, messages, max_tokens, gptcli=gptcli2)

        max_response = min(8191 - gptcli.estimate_messages_num_tokens(messages, model=model), max_tokens)

        total_tokens = 0
        total_uncached_tokens = 0
        response = gptcli.chat_completion(
            model=model,
            messages=messages,
            temperature=0,
            max_tokens=max_response,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        total_tokens += response["usage"]["total_tokens"]
        total_uncached_tokens += response["usage"]["uncached_total_tokens"]
        result_text = response["choices"][0]["message"]["content"]
        if total_uncached_tokens != 0 and not self.quiet:
            print("total={} uncached_total={}".format(total_tokens, total_uncached_tokens))
        response["input"] = messages.copy()
        return result_text, response


class MultiAgentGroup:
    def __init__(
        self,
        config,
        doc_edits,
        model_name,
        paper_chunk_size=4 * (2**10),
        max_tokens=800,
        prompts=None,
        quiet=False,
        use_history_pruning=False,
        doc_mods=None,
        master_chunk_type="normal",
        color_format="ansi",
        taxonomy="",
        extra_bots=None,
        raw_paper_chunks=None,
    ):
        self.config = config
        self.doc_edits = doc_edits
        self.model_name = model_name
        self.paper_chunk_size = paper_chunk_size
        self.max_tokens = max_tokens
        self.prompts = prompts
        self.quiet = quiet
        self.use_history_pruning = use_history_pruning
        self.master_chunk_type = master_chunk_type
        self.color_format = color_format
        self.taxonomy = taxonomy
        self.extra_bots = extra_bots or []

        if raw_paper_chunks is None:
            with aries.util.gpt3.Gpt3CacheClient(config["gpt3_cache_db_path"]) as gptcli:
                self.paper_chunks, _ = make_chunked_paper_diff(
                    doc_edits, paper_chunk_size, gptcli, mode="source", model_name="gpt-4", color_format="none", doc_mods=doc_mods
                )
        else:
            self.paper_chunks = raw_paper_chunks

        self.bot_experts = []
        self.init_swarm()

    def init_swarm(self):
        CHAT_COLORS = [
            "magenta",
            "blue",
            "yellow",
            "green",
            "cyan",
            "red",
            "strong-blue",
            "strong-yellow",
            "strong-green",
            "strong-cyan",
            "strong-red",
        ]
        self.bot_experts = []
        self.extra_bot_experts = []

        agent_idx = 0
        if self.master_chunk_type == "none":
            bot = GptChatBot(
                self.config["gpt3_cache_db_path"],
                model=self.model_name,
                system_prompt=self.prompts["master_system_prompt"],
                max_tokens=self.max_tokens,
            )
            bot.paper_chunk = ""
            bot.agent_type = "leader"
            bot.agent_name = "Agent {}".format(agent_idx)
            bot.chat_color = CHAT_COLORS[agent_idx % len(CHAT_COLORS)]
            self.bot_experts.append(bot)

        for idx, chunk in enumerate(self.paper_chunks):
            agent_idx = len(self.bot_experts)
            if agent_idx == 0:
                bot = GptChatBot(
                    self.config["gpt3_cache_db_path"],
                    model=self.model_name,
                    system_prompt=self.prompts["master_system_prompt"],
                    max_tokens=self.max_tokens,
                )
                bot.agent_type = "leader"
            else:
                bot = GptChatBot(
                    self.config["gpt3_cache_db_path"],
                    model=self.model_name,
                    system_prompt=self.prompts["worker_system_prompt"],
                    max_tokens=self.max_tokens,
                )
                bot.agent_type = "worker"
            bot.paper_chunk = chunk
            bot.agent_name = "Agent {}".format(agent_idx)
            bot.chat_color = CHAT_COLORS[agent_idx]
            self.bot_experts.append(bot)

        for _, bot_prompts in enumerate(self.extra_bots):
            agent_idx = len(self.bot_experts)
            bot = GptChatBot(
                self.config["gpt3_cache_db_path"],
                model=self.model_name,
                system_prompt=bot_prompts["system_prompt"],
                max_tokens=self.max_tokens,
            )
            bot.paper_chunk = ""
            bot.agent_type = "extra"
            bot.extra_prompts = bot_prompts
            bot.agent_name = "Agent {}".format(agent_idx)
            bot.chat_color = CHAT_COLORS[agent_idx]
            self.extra_bot_experts.append(bot)
            self.bot_experts.append(bot)

        # print(bot_experts[0].paper_chunk)
        for idx, bot in enumerate(self.bot_experts):
            other_agents = [x.agent_name for x in self.bot_experts if x.agent_name != bot.agent_name]
            if bot.agent_type == "leader":
                rt0, resp0 = bot.chat(
                    self.prompts["master_chunk_prompt"].format(
                        source_paper_chunk=bot.paper_chunk,
                        num_agents=len(self.bot_experts),
                        agent_name=bot.agent_name,
                        other_agent_names=", ".join(other_agents),
                        taxonomy=self.taxonomy,
                    ),
                    max_tokens=self.max_tokens,
                )
            elif bot.agent_type == "worker":
                rt1, resp1 = bot.chat(
                    self.prompts["worker_chunk_prompt"].format(
                        source_paper_chunk=bot.paper_chunk,
                        num_agents=len(self.bot_experts),
                        agent_name=bot.agent_name,
                        other_agent_names=", ".join(other_agents),
                        taxonomy=self.taxonomy,
                    ),
                    max_tokens=self.max_tokens,
                )
            elif bot.agent_type == "extra":
                rt1, resp1 = bot.chat(
                    bot.extra_prompts["chunk_prompt"].format(
                        source_paper_chunk=bot.paper_chunk,
                        num_agents=len(self.bot_experts),
                        agent_name=bot.agent_name,
                        other_agent_names=", ".join(other_agents),
                        taxonomy=self.taxonomy,
                    ),
                    max_tokens=self.max_tokens,
                )
            else:
                raise ValueError("Unknown agent type")

    def prune_history(self, messages, skip_n=0, replace_content=None, reverse_skip_n=0):
        pruned_messages = []
        skipped_n = 0
        for msg_idx, msg in enumerate(messages):
            # Experimental: prune messages that had to be retransmitted (usually because "SEND MESSAGE" was in the wrong place)
            if (
                msg["role"] == "assistant"
                and (msg["content"].startswith("SEND MESSAGE:") or msg["content"].startswith("SEND FULL MESSAGE:"))
                and msg_idx > 0
            ):
                # This looks like a retransmission; let's check by seeing if
                # the non-sent content of the previous message uses roughly the
                # same tokens
                prev_assistant_msg = None
                prev_assistant_msg_idx = None
                for prev_msg_idx, prev_msg in reversed(list(enumerate(pruned_messages))):
                    if prev_msg["role"] == "assistant" and prev_msg_idx >= len(pruned_messages) - 2:
                        prev_assistant_msg = prev_msg
                        prev_assistant_msg_idx = prev_msg_idx
                        break
                if prev_assistant_msg is not None and "SEND MESSAGE" in prev_assistant_msg["content"]:
                    prev_msg_content = prev_assistant_msg["content"][: prev_assistant_msg["content"].index("SEND MESSAGE")]
                    prev_msg_tokens = collections.Counter(prev_msg_content.split())
                    msg_tokens = collections.Counter(msg["content"].split())
                    if aries.util.data.counter_jaccard(prev_msg_tokens, msg_tokens) > 0.9:
                        new_msg = prev_assistant_msg.copy()
                        new_msg["content"] = (
                            prev_msg_content[prev_assistant_msg["content"].index("SEND MESSAGE") :]
                            + "\n\n"
                            + msg["content"][msg["content"].index(":") + 1 :]
                        )
                        while len(pruned_messages) > prev_assistant_msg_idx:
                            pruned_messages.pop()
                        pruned_messages.append(new_msg)
                        continue

            if not (msg["role"] == "system" and msg["content"].startswith("Message from Agent") and len(messages) - msg_idx > reverse_skip_n):
                pruned_messages.append(msg)
                continue
            elif skipped_n < skip_n:
                skipped_n += 1
                pruned_messages.append(msg)
                continue
            elif replace_content:
                new_msg = msg.copy()
                new_msg["content"] = replace_content
                pruned_messages.append(new_msg)
                continue
        return pruned_messages

    def _self_colorprint(self, *args, **kwargs):
        # print_fn = print
        print_fn = kwargs.pop("print_fn", logger.info)
        if "form" in kwargs:
            if kwargs["form"] == "html":

                def print_fn(x):
                    # display(IPython.display.HTML(x.replace('\n', '<br>')))
                    logger.info(x.replace("\n", "<br>"))

        print_fn(colorify(*args, **kwargs))

    def ask_swarm_question(self, question, pre_prompt="", stop_strings=None):
        PRUNE_STRING = "This doesn't seem relevant to me, so I will stand by for further instructions."

        # For now we just assume bot 0 is the "main" one
        rt, resp = self.bot_experts[0].chat("{}{}".format(pre_prompt, question))

        old_rts = {rt}
        while True:
            if not self.quiet:
                self._self_colorprint(rt, color=(self.bot_experts[0].chat_color or "red"), form=self.color_format)
            if stop_strings is not None and any(x in rt for x in stop_strings):
                break
            # msglines = [x for x in rt.split('\n') if x.startswith('SEND MESSAGE')]
            if "SEND FULL MESSAGE" in rt:
                msgline = rt.replace("SEND FULL MESSAGE", "")
            else:
                # msgidx = rt.find("SEND MESSAGE:")
                msgidx = rt.find("SEND MESSAGE")
                # if len(msglines) == 0:
                # if msgidx == -1 or (msgidx > 0 and rt[msgidx-1] != '\n'):
                if msgidx == -1 or (msgidx > 0 and rt[msgidx - 1] not in ("\n", " ")):
                    break
                # For now we just consider one message
                # msgline = msglines[0]
                msgline = rt[msgidx + len("SEND MESSAGE: ") :]
                # msgline = rt.replace('SEND MESSAGE: ', '')
            rt2s = []
            for bot_idx, bot in enumerate(self.bot_experts[1:]):
                if self.use_history_pruning:
                    if bot.agent_type == "worker":
                        # bot.messages = bot.messages[:3]
                        msgs = bot.messages
                        bot.messages = bot.messages[:5]
                        if len(msgs) >= 7:
                            bot.messages[3:5] = msgs[5:7]
                    elif bot.agent_type == "extra":
                        new_msgs = []
                        for msg_idx, msg in enumerate(bot.messages):
                            if (msg["role"] == "assistant" and msg["content"].strip() == PRUNE_STRING) or (
                                msg_idx < len(bot.messages) - 1
                                and bot.messages[msg_idx + 1]["content"].strip() == PRUNE_STRING
                                and bot.messages[msg_idx + 1]["role"] == "assistant"
                            ):
                                continue
                            new_msgs.append(msg)
                        bot.messages = new_msgs
                rt2, resp2 = bot.chat("Message from Agent 0: {}".format(msgline), role="system", max_tokens=self.max_tokens)
                if not self.quiet:
                    self._self_colorprint(rt2, color=(bot.chat_color or "blue"), form=self.color_format)
                # rt2s.append({'agent': bot.agent_name, 'replytext': rt2})
                if rt2.strip() == PRUNE_STRING:
                    continue
                    if bot.agent_name in msgline:
                        rt2, resp2 = bot.chat(
                            "Your name is in the message; are you sure it is not addressed to you?  If it is not for you, repeat your previous message.  If it is for you, write your response."
                        )
                        if not self.quiet:
                            self._self_colorprint(rt2, color=(bot.chat_color or "blue"), form=self.color_format)
                        if rt2.strip() == PRUNE_STRING:
                            continue
                    else:
                        continue
                rt2s.append({"agent": bot.agent_name, "msg": "Message from {}: {}".format(bot.agent_name, rt2.replace("SEND MESSAGE: ", ""))})
            rtmsg = "\n\n".join(x["msg"] for x in rt2s)
            if rtmsg.strip() == "":
                rtmsg = "No messages were received from any agent; you may need to reword and double-check your message so that the agents know who your message is intended for."
            rtmsg += "\n\nSystem note: remember that if you need to send a message back, you need to prepend SEND MESSAGE"
            rt, resp = self.bot_experts[0].chat(rtmsg, role="system", max_tokens=self.max_tokens)
            while rt in old_rts:
                print("DUP ERROR: {}".format(rt))
                rt2, resp = self.bot_experts[0].chat(
                    "Error: you tried to send exactly the same message as before.  You have already received a response to that message from all agents.  Note: If the conversation is over, you should stop sending messages.",
                    role="system",
                )
                if rt2 == rt:
                    print("SUPER DUP ERROR: {}".format(rt2))
                    return rt2, resp
                rt = rt2

            if self.use_history_pruning:
                self.bot_experts[0].messages = self.prune_history(
                    self.bot_experts[0].messages,
                    skip_n=0,
                    replace_content="[ The agents responded to you, but their messages have been pruned from the history. Your response below was created based on their real messages. ]",
                )
            old_rts.add(rt)
        return rt, resp


def make_chunked_paper_diff(
    doc_edits, chunk_size, gptcli, source_start_paragraph=0, mode="diff", model_name="gpt-4", color_format="none", doc_mods=None, skip_abstract=False
):
    if mode == "diff":
        full_diff_string, edits_by_id = doc_edits.make_paper_diff_string(
            color_format=color_format,
            print_ids_only=True,
            return_edit_ids=True,
        )
    elif mode == "source":
        del_paras = set()
        mod_paras = dict()
        if doc_mods is not None:
            del_paras = doc_mods["delete_ids"]
            mod_paras = index_by(doc_mods["paragraph_modifications"], "paragraph_id", one_to_one=True)
        full_diff_string = ""
        edits_by_id = dict()

        def format_source_text(para_text, para_id, section_name="unknown"):
            s = ""
            s += "section: {}".format(section_name)
            s += "\nparagraph id: {}".format(para_id)
            s += "\n" + para_text
            s += "\n\n"
            return s

        display_idx = -1
        for edit_idx, edit in enumerate(doc_edits.iter_source_edits()):
            display_idx += 1
            if edit_idx < source_start_paragraph:
                continue
            if edit.get_source_text().isspace():
                continue
            edits_by_id[edit_idx] = edit

            section_name = "unknown"
            if len(edit.source_idxs) != 0:
                section_name = doc_edits.s2orc1["pdf_parse"]["body_text"][edit.source_idxs[0]]["section"]

            if edit.get_source_text().strip() == "":
                continue

            if edit_idx in del_paras:
                continue

            para_text = edit.get_source_text()
            if edit_idx in mod_paras:
                para_text = mod_paras[edit_idx]["new_text"]

            full_diff_string += format_source_text(para_text, display_idx, section_name=section_name)
        if not skip_abstract:
            abstract_text = doc_edits.s2orc1["pdf_parse"]["abstract"][0]["text"].strip()
            if 9999 in mod_paras:
                abstract_text = mod_paras[9999]["new_text"]
            if 9999 not in del_paras and abstract_text != "":
                full_diff_string = format_source_text(abstract_text, 9999, section_name="Abstract") + full_diff_string
    else:
        raise ValueError("Unknown mode {}".format(mode))

    para_chunks = full_diff_string.split("\n\n")

    diff_chunks = []
    cur_chunk = []
    cur_chunk_len = 0
    # Note: we don't account for individual paras being bigger than
    # chunk_size; that probably never happens anyway
    for para_chunk in para_chunks:
        # Add 2 for the stripped \n\n
        new_chunk_len = gptcli.estimate_num_tokens(para_chunk, model_name) + 2
        if cur_chunk_len + new_chunk_len > chunk_size:
            diff_chunks.append("\n\n".join(cur_chunk))
            cur_chunk = []
            cur_chunk_len = 0
        cur_chunk.append(para_chunk)
        cur_chunk_len += new_chunk_len

    if len(cur_chunk) != 0:
        diff_chunks.append("\n\n".join(cur_chunk))

    return diff_chunks, edits_by_id
