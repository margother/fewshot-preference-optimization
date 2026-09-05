"""
Adapts our own build_personalization_dataset.py output (train.json: rows
with user_id/prompt/chosen/rejected, plain strings) into the SAME
{prompt: {responses, pairs, sft_target}} shape every one of the original
repo's own dataset functions (get_elix/get_review/get_roleplay, in
../directpreferenceoptimization/preference_datasets.py) produce -- see
that file's own shots_to_text() for the exact few-shot text format.

Mirrors get_elix's own windowing/reshuffling mechanism EXACTLY (see chat:
"we need to absolutely avoid refining their method" -- this is a faithful
reproduction of their own mechanism applied to our data, not a redesign):
    for _ in range(2 * (num_shots + 1)):
        for each user's own row pool (reshuffled fresh EVERY pass):
            walk non-overlapping windows of (num_shots + 2) rows:
                rows[0:num_shots]   -> few-shot context
                rows[num_shots]     -> this block's target question
                rows[num_shots + 1] -> UNUSED (their own off-by-one gap,
                                        present in get_elix/get_review/
                                        get_roleplay too -- kept as-is, not
                                        "fixed", per the same faithfulness
                                        principle)
This is data augmentation via repeated re-shuffled re-chunking, NOT
multiple training epochs -- see chat: "is this like several epochs?!" ->
it runs once, at dataset-construction time, before the trainer's own
n_epochs loop ever starts.

num_shots=4 (switched from 8 -- see chat: "updates?" thread, job 328679's
own real OOM in the IPO stage: concatenated_forward processes chosen+
rejected together AND needs a frozen reference model, both roughly
doubling memory versus SFT even at microbatch=1, with no smaller unit to
fall back to. Reducing context length at the SOURCE via num_shots is
cleaner than fighting the trainer's own memory mechanics, and num_shots=4
is STILL one of their own real conventions, not a new number -- it's
elix's own default, just not review/roleplay's), include_disprefferred=True
-- matches the original repo's own dataset registrations in get_dataset().

Usage (from ../directpreferenceoptimization/preference_datasets.py's own
get_dataset(), after the registration import added there):
    data = get_personalllm(train_path, split="train")
"""
import json
import random
from collections import defaultdict
from typing import Dict, List, Union

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "directpreferenceoptimization"))
from preference_datasets import shots_to_text


def get_personalllm(
    split: str = "train",
    silent: bool = False,
    # PERSONALLLM_NUM_SHOTS env var override -- see chat: "can we try
    # num_shots=2?" (a memory-budget-fit search for llama/n10, one GPU) --
    # same threading pattern as PERSONALLLM_TRAIN_PATH/_TEST_PATH already
    # uses (get_dataset()'s own call site, preference_datasets.py:356,
    # never passes num_shots at all, so this was the ONLY way to override
    # it per-run without hand-editing this hardcoded default -- which is
    # what qwen's own 8->4 switch actually did, a real regression risk if
    # ever hand-edited again and forgotten: this default is SHARED across
    # every model/tree, qwen's own already-registered num_shots=4 results
    # depend on it staying 4 unless a launcher explicitly opts out).
    # getattr-free os.environ.get(...) default, evaluated once at import
    # time (the launcher exports the env var before python starts, same as
    # every other PERSONALLLM_* override here) -- default "4" preserves
    # qwen's exact existing behavior byte-for-byte when unset.
    num_shots: int = int(os.environ.get("PERSONALLLM_NUM_SHOTS", 4)),
    include_disprefferred: bool = True,
) -> Dict[str, Dict[str, Union[List, str]]]:
    # train/test path comes from an env var, not a get_dataset()-threaded
    # param -- get_dataset()'s own signature (name, split, silent, cache_dir,
    # include_level) is shared by every dataset function and called from
    # multiple sites in preference_datasets.py; adding a new parameter
    # there would ripple outward into every caller for a personalllm-only
    # need. An env var, set by our own training launcher before calling
    # train.py, is the minimal way to thread this through without touching
    # any of their own function signatures.
    #
    # split-aware (PERSONALLLM_TEST_PATH for split="test") -- previously
    # this ALWAYS read PERSONALLLM_TRAIN_PATH regardless of split, a real
    # latent bug: BasicTrainer.__init__ (trainers.py) already builds its own
    # self.eval_iterator via get_batch_iterator(..., split='test', ...) --
    # once real training reaches eval_every (10000, first hit mid-epoch on
    # the ~36k-example augmented set), it would have silently evaluated on
    # TRAIN data. Caught before any real training run reached that point
    # (see chat: "can you add this [test accuracy]... after the training
    # job").
    if split == "test":
        # .get() with a fallback, not a hard KeyError -- job 328737's own
        # real crash: mp.spawn's rank-1 FSDP worker raised KeyError on
        # PERSONALLLM_TEST_PATH even though the export is correctly placed
        # before the python call and PERSONALLLM_TRAIN_PATH has inherited
        # reliably across every prior multi-process attempt (a genuine, if
        # unexplained, env-inheritance flake specific to that one var/rank).
        # Harmless to fall back here -- with eval_every set far past any
        # real epoch's example count (see train_seen.sh/train_seen_sanity_
        # check.sh's own comments), this eval split is never actually read
        # during training; only compute_test_accuracy.py's own separate,
        # single-process (no mp.spawn) invocation needs the real value, and
        # it always sets the env var itself before importing this module.
        data_path = os.environ.get("PERSONALLLM_TEST_PATH") or os.environ["PERSONALLLM_TRAIN_PATH"]
    else:
        data_path = os.environ["PERSONALLLM_TRAIN_PATH"]
    with open(data_path) as f:
        rows = json.load(f)

    by_user = defaultdict(list)
    for r in rows:
        by_user[r["user_id"]].append(r)

    data = defaultdict(lambda: defaultdict(list))
    for _ in range(2 * (num_shots + 1)):
        for user_id, user_rows in by_user.items():
            user_rows = user_rows[:]  # don't mutate the shared list -- reshuffled fresh each pass.
            random.shuffle(user_rows)
            index_df = 0
            while index_df < len(user_rows):
                start = index_df
                end = index_df + (num_shots + 2)  # get N few-shot examples and 1 prompt -- see module docstring's own off-by-one note.
                if end > len(user_rows):
                    break
                block = user_rows[start:end]

                fewshot_prompts = [block[i]["prompt"] for i in range(num_shots)]
                fewshot_preferred = [block[i]["chosen"] for i in range(num_shots)]
                fewshot_dispreferred = [block[i]["rejected"] for i in range(num_shots)]

                final_question = block[num_shots]["prompt"]
                final_chosen = block[num_shots]["chosen"]
                final_rejected = block[num_shots]["rejected"]

                prompt, chosen, rejected = shots_to_text(
                    fewshot_prompts=fewshot_prompts,
                    fewshot_preferred=fewshot_preferred,
                    fewshot_dispreferred=fewshot_dispreferred,
                    final_question=final_question,
                    final_chosen=final_chosen,
                    final_rejected=final_rejected,
                    include_dispreferred=include_disprefferred,
                    add_persona=False,
                )
                responses = [chosen, rejected]
                n_responses = len(data[prompt]["responses"])
                data[prompt]["pairs"].append((n_responses, n_responses + 1))
                data[prompt]["responses"].extend(responses)
                data[prompt]["sft_target"].extend([chosen, rejected])

                index_df = end
    return data
