"""PRISM sibling of ../fspo_personalllm/personalllm_data.py -- see chat:
"whatever launcher is going to be needed to replicate the personalllm
experiment on prism" (PRISM pipeline plan, step 8). A duplicate, not a
shared import (same "8 shared utilities deliberately duplicated" split
convention dpo_LRF/data/loaders/{personalllm,prism}.py already established
in the main env) -- own env var names (PRISM_TRAIN_PATH/PRISM_TEST_PATH/
PRISM_NUM_SHOTS, not the PERSONALLLM_* ones) so a PRISM run and a
personalllm run never silently clobber each other's paths if launched
concurrently on the same node.

Everything else is IDENTICAL to personalllm_data.py's own get_personalllm
-- adapts our own build_personalization_dataset.py output (train.json:
rows with user_id/prompt/chosen/rejected, plain strings; PRISM's own rows
have this EXACT shape too, see dpo_LRF/prism/build_personalization_
dataset.py's own _examples_to_rows) into the SAME {prompt: {responses,
pairs, sft_target}} shape every one of the original repo's own dataset
functions produce. Mirrors get_elix's own windowing/reshuffling mechanism
EXACTLY (see chat: "we need to absolutely avoid refining their method") --
a faithful reproduction of their own mechanism applied to our data, not a
redesign. This is data augmentation via repeated re-shuffled re-chunking,
NOT multiple training epochs -- runs once, at dataset-construction time.

num_shots=4/include_disprefferred=True -- same qwen/n10 memory-budget
choice personalllm_data.py's own module docstring explains (elix's own
default, not a new number) -- reused here since it's a model/GPU-memory
constraint, not a dataset-specific one.

Usage (from ../directpreferenceoptimization/preference_datasets.py's own
get_dataset(), after the registration import added there):
    data = get_prism(split="train")
"""
import json
import random
from collections import defaultdict
from typing import Dict, List, Union

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "directpreferenceoptimization"))
from preference_datasets import shots_to_text


def get_prism(
    split: str = "train",
    silent: bool = False,
    num_shots: int = int(os.environ.get("PRISM_NUM_SHOTS", 4)),
    include_disprefferred: bool = True,
) -> Dict[str, Dict[str, Union[List, str]]]:
    if split == "test":
        # .get() with a fallback -- same env-inheritance-flake defense
        # personalllm_data.py's own get_personalllm already applies (job
        # 328737's own real crash under mp.spawn), applied here on the
        # same principle even though not yet observed for PRISM.
        data_path = os.environ.get("PRISM_TEST_PATH") or os.environ["PRISM_TRAIN_PATH"]
    else:
        data_path = os.environ["PRISM_TRAIN_PATH"]
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
