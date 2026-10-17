"""Shared dataset adapter for personalllm/prism/community_alignment/roleplay_pairs -- see chat:
"could we rather use a generic adaptor as in the other baseline?" (same naming-inertia pattern
already fixed for LoRe/generic and Personalized_RLHF/prlhf_generic -- this one started as
personalllm_data.py, then got a byte-identical PRISM sibling (prism_data.py) that community_
alignment/roleplay_pairs already reused via an env_prefix parameter; this file finishes that
consolidation into ONE function instead of two near-duplicate files).

Adapts our own build_personalization_dataset.py output (train.json: rows with user_id/prompt/
chosen/rejected, plain strings; every dataset family's own rows have this exact shape) into the
SAME {prompt: {responses, pairs, sft_target}} shape every one of the original repo's own dataset
functions (get_elix/get_review/get_roleplay, in ../directpreferenceoptimization/
preference_datasets.py) produce -- see that file's own shots_to_text() for the exact few-shot
text format.

Mirrors get_elix's own windowing/reshuffling mechanism EXACTLY (see chat: "we need to absolutely
avoid refining their method" -- a faithful reproduction of their own mechanism applied to our
data, not a redesign):
    for _ in range(2 * (num_shots + 1)):
        for each user's own row pool (reshuffled fresh EVERY pass):
            walk non-overlapping windows of (num_shots + 2) rows:
                rows[0:num_shots]   -> few-shot context
                rows[num_shots]     -> this block's target question
                rows[num_shots + 1] -> UNUSED (their own off-by-one gap, present in get_elix/
                                        get_review/get_roleplay too -- kept as-is, not "fixed",
                                        per the same faithfulness principle)
This is data augmentation via repeated re-shuffled re-chunking, NOT multiple training epochs --
it runs once, at dataset-construction time, before the trainer's own n_epochs loop ever starts.

num_shots=4, include_disprefferred=True -- elix's own convention (not review/roleplay's), kept
as the shared default for every one of our own registrations; PersonalLLM/qwen/n3's own already-
registered result depends on this staying 4 unless a launcher explicitly opts out via
{PREFIX}_NUM_SHOTS.

env_prefix (REQUIRED, no default -- every call site below passes it explicitly): reads
{prefix}_TRAIN_PATH / {prefix}_TEST_PATH / {prefix}_NUM_SHOTS. Own, distinct env var names per
dataset so two datasets' runs launched concurrently on the same node never clobber each other's
paths through one shared global name.

Test split: split-aware (split="test" reads {prefix}_TEST_PATH) -- previously PersonalLLM's own
version of this always read {prefix}_TRAIN_PATH regardless of split, a real latent bug:
BasicTrainer.__init__ (trainers.py) already builds its own self.eval_iterator via
get_batch_iterator(..., split='test', ...) -- once real training reaches eval_every (10000,
first hit mid-epoch on the ~36k-example augmented set), it would have silently evaluated on
TRAIN data. Caught before any real training run reached that point. The test-path read also uses
.get() with a fallback, not a hard KeyError -- job 328737's own real crash: mp.spawn's rank-1
FSDP worker raised KeyError on {prefix}_TEST_PATH even though the export is correctly placed
before the python call and {prefix}_TRAIN_PATH has inherited reliably across every prior multi-
process attempt (a genuine, if unexplained, env-inheritance flake specific to that one var/rank).
Harmless to fall back here -- with eval_every set far past any real epoch's example count, this
eval split is never actually read during training; only compute_test_accuracy.py's own separate,
single-process (no mp.spawn) invocation needs the real value, and it always sets the env var
itself before importing this module.

Usage (from ../directpreferenceoptimization/preference_datasets.py's own get_dataset(), after
the registration import added there):
    data = get_fewshot_dataset(split="train", env_prefix="PERSONALLLM")
"""
import json
import random
from collections import defaultdict
from typing import Dict, List, Optional, Union

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "directpreferenceoptimization"))
from preference_datasets import shots_to_text


def get_fewshot_dataset(
    split: str,
    env_prefix: str,
    silent: bool = False,
    num_shots: Optional[int] = None,
    include_disprefferred: bool = True,
) -> Dict[str, Dict[str, Union[List, str]]]:
    if num_shots is None:
        num_shots = int(os.environ.get(f"{env_prefix}_NUM_SHOTS", 4))
    if split == "test":
        data_path = os.environ.get(f"{env_prefix}_TEST_PATH") or os.environ[f"{env_prefix}_TRAIN_PATH"]
    else:
        data_path = os.environ[f"{env_prefix}_TRAIN_PATH"]
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
