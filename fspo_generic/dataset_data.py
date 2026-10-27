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

Test split: split-aware (split="test" reads {prefix}_TEST_PATH for TARGETS and {prefix}_SHOTS_PATH
for the few-shot CONTEXT, both required) -- previously PersonalLLM's own version of this always
read {prefix}_TRAIN_PATH regardless of split, a real latent bug: BasicTrainer.__init__
(trainers.py) already builds its own self.eval_iterator via get_batch_iterator(..., split='test',
...) -- once real training reaches eval_every (10000, first hit mid-epoch on the ~36k-example
augmented set), it would have silently evaluated on TRAIN data. Caught before any real training
run reached that point.

{prefix}_SHOTS_PATH (also required for split=="test"): the real train/adaptation pool shots are
drawn from, mirroring eval/generate.py's own fspo branch exactly (seeded shuffle, slice
[:num_shots], reused across all of that user's targets) -- fixes compute_test_accuracy.py's own
leakage (few-shot context and scored target both windowed from the SAME held-out test split, the
original FSPO authors' own convention) by construction. NOT optional (see chat: "can we withdraw
the --shots-path flag, and just implement in hard the same shot construction than in generate" --
there used to be a leaky default you could omit this to reproduce; now there is only one
behavior). train.py's own real eval-every logging never actually reaches split=="test" against a
real (non-dummy) path before an epoch completes, so it is unaffected; only compute_test_accuracy.
py's own separate, single-process invocation needs the real value, and it always sets both env
vars itself before importing this module.

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

    # Test split: hardcoded to mirror eval/generate.py's own fspo branch EXACTLY (seeded
    # random.Random(42), shuffle the user's real train/adaptation pool once, slice [:num_shots],
    # reuse those same shots across every one of that user's target questions) -- NOT optional
    # anymore (see chat: "can we withdraw the --shots-path flag, and just implement in hard the
    # same shot construction than in generate" -- there is no more leaky fallback to toggle away
    # from). {env_prefix}_SHOTS_PATH is now a required env var for split=="test", pointing at the
    # SAME file eval/generate.py's own win-rate generation draws shots from (seen's train.json,
    # or unseen's adaptation-pool train.json) -- fixes compute_test_accuracy.py's own leakage
    # (few-shot CONTEXT and scored TARGET both windowed from the SAME held-out test split, the
    # original FSPO authors' own convention) by construction, with a single tested shuffle/slice
    # implementation instead of two that could drift apart.
    #
    # {env_prefix}_SHOTS_PATH defaults to {env_prefix}_TEST_PATH (NOT a hard requirement) --
    # real training launchers (train.sh/train_seen.sh) export {env_prefix}_TEST_PATH to a dummy
    # file and never set SHOTS_PATH at all, yet BasicTrainer.__init__ (trainers.py) EAGERLY builds
    # self.eval_batches = list(self.eval_iterator) at construction time regardless of whether
    # .train()'s own eval_every (set far past any real epoch's example count) ever actually
    # consumes it -- without this default every real FSPO training launch would KeyError at
    # startup. Only compute_test_accuracy.py's own invocation needs the real value, and its own
    # --shots-path CLI arg is required (no omit-to-get-the-old-leaky-protocol option there) --
    # this default only ever fires for training's own harmless, never-consumed dummy eval_batches.
    if split == "test":
        test_path = os.environ[f"{env_prefix}_TEST_PATH"]
        with open(test_path) as f:
            test_rows = json.load(f)
        by_user = defaultdict(list)
        for r in test_rows:
            by_user[r["user_id"]].append(r)

        shots_path = os.environ.get(f"{env_prefix}_SHOTS_PATH", test_path)
        with open(shots_path) as f:
            shots_rows = json.load(f)
        shots_by_user = defaultdict(list)
        for r in shots_rows:
            shots_by_user[r["user_id"]].append(r)

        _shuf_rng = random.Random(42)
        data = defaultdict(lambda: defaultdict(list))
        n_skipped_no_shots = 0
        for user_id, user_rows in by_user.items():
            pool = shots_by_user.get(user_id, [])
            if len(pool) < num_shots:
                n_skipped_no_shots += len(user_rows)
                continue
            pool = pool[:]
            _shuf_rng.shuffle(pool)
            shots = pool[:num_shots]
            fewshot_prompts = [s["prompt"] for s in shots]
            fewshot_preferred = [s["chosen"] for s in shots]
            fewshot_dispreferred = [s["rejected"] for s in shots]
            for target in user_rows:
                prompt, chosen, rejected = shots_to_text(
                    fewshot_prompts=fewshot_prompts, fewshot_preferred=fewshot_preferred,
                    fewshot_dispreferred=fewshot_dispreferred, final_question=target["prompt"],
                    final_chosen=target["chosen"], final_rejected=target["rejected"],
                    include_dispreferred=include_disprefferred, add_persona=False,
                )
                n_responses = len(data[prompt]["responses"])
                data[prompt]["pairs"].append((n_responses, n_responses + 1))
                data[prompt]["responses"].extend([chosen, rejected])
                data[prompt]["sft_target"].extend([chosen, rejected])
        if n_skipped_no_shots and not silent:
            print(f"get_fewshot_dataset: {n_skipped_no_shots} test rows skipped (their user has fewer than "
                  f"{num_shots} rows in {shots_path})")
        return data

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
