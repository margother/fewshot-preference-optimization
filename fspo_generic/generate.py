"""
Generates {uid: {prompt: text}}-shaped output for a trained FSPO (SFT+IPO)
checkpoint (train_seen.sh's own output) -- conforms to the SAME JSON
contract dpo_LRF/eval/generate.py's own output does (see that script's own
docstring: "any script producing {uid: {prompt: generated_text}} in this
exact shape is a valid input to eval/downstream_eval.py"), so this
checkpoint gets scored by the SAME shared scoring script every other
method uses, without any FSPO-specific eval code. Mirrors ../../
Personalized_RLHF/prlhf_personalllm/generate.py's own structure exactly
(module docstring, argparse conventions, resumable output).

Few-shot context: uses num_shots real (question, chosen, rejected)
examples from the SAME raw --train-path rows training itself reads
(get_personalllm()'s own PERSONALLLM_TRAIN_PATH), grouped by user_id, then
the user's own single held-out test prompt (from --dataset-path, the
resolved eval file) as the final question. NOT the resolved dataset's own
"train_examples" field -- that shape only carries {prompt, chosen} (built
for RAG's own simpler in-context examples, no preference PAIR needed
there), no "rejected" at all, confirmed by reading a real resolved
dataset file rather than assumed (a first version of this script assumed
train_examples had "rejected" too and would have KeyError'd on the very
first user). No retrieval/similarity selection (see chat: "we need to
absolutely avoid refining their method" -- their own get_elix/get_review/
get_roleplay windowing has no retrieval step either) -- takes the FIRST
num_shots rows of that user's own pool.

max_new_tokens=1024 (NOT their own trainers.py's own get_batch_samples,
which caps by max_length instead) -- see chat: "can we make sure that the
generation length 1024 is hardcoded across all methods to avoid silent-
drift?". This eval/generation step is OUR OWN harness code (same as it
already is for every other method, P-RLHF included), not part of "their
method" (their method is the SFT+IPO training procedure) -- using the
SAME response-length budget as every other method keeps the downstream
reward-model comparison fair, regardless of what convention their own
internal during-training eval sampling uses.

do_sample=True (NOT a --decoding flag with a greedy option, unlike
P-RLHF's own generate.py) -- their own trainers.py's own get_batch_samples
is the ONLY generation call in their entire codebase, and it always uses
do_sample=True, no greedy variant shown anywhere. Faithful reproduction
means mirroring that, not inventing a greedy option they never had.

Usage:
    python generate.py \\
        --archive /path/to/personalllm_seen_ipo_TIMESTAMP/LATEST/policy.pt \\
        --dataset-path /path/to/resolved_eval_dataset.json \\
        --output-path /path/to/generations_alpha0.01.json
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "directpreferenceoptimization"))
from preference_datasets import shots_to_text

# MUST match dpo_LRF/personalllm/gob_hparams.py's own GOB_MAX_NEW_TOKENS
# (also 1024) -- see module docstring. Manually kept-in-sync literal, same
# reasoning as prlhf_personalllm/generate.py's own identical constant --
# this repo's own env_fspo is deliberately decoupled from the dpo_LRF
# package too.
MAX_NEW_TOKENS = 1024

# Matches the literal training-target prefix shots_to_text() (preference_
# datasets.py:57-61) prepends to every chosen/rejected completion -- "##
# Preferred Response \n" when include_dispreferred=True (this project's own
# --include-dispreferred default), "## Response \n" otherwise. The model
# faithfully reproduces it at generation time too (confirmed: 481/500 real
# generations from a completed alpha0.01 run started with this exact
# string) since the prompt handed to model.generate() stops right before
# it, so the model has to emit the header itself to continue the pattern.
# NOT a bug in their training -- it's a direct, correct consequence of the
# target format -- but it has no business in text that then gets scored by
# a reward model expecting a plain answer, or read by a human. See chat:
# "how to handle this problem cleanly for future generations? I think we
# already had this problem for ICL?" -- ICL's own analogous fix
# (dpo_LRF/prompts/rag_prompts.py's build_icl_system_prompt) puts the
# few-shot exemplars in a real chat-template system turn instead, an
# unambiguous role boundary the model can't blur into its own answer. That
# fix can't transfer here: FSPO's own method IS raw-text few-shot
# continuation with no chat template at all (see chat: "we need to
# absolutely avoid refining their method") -- so this strips the leaked
# header downstream of generation instead, right where the resumable
# output gets written, rather than changing how the prompt is built.
_RESPONSE_HEADER_RE = re.compile(r"^(?:#+\s*\n+)*##\s*(?:Preferred )?Response\s*\n+")


def _strip_response_header(text: str) -> str:
    return _RESPONSE_HEADER_RE.sub("", text, count=1).strip()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--archive", required=True, help="train_seen.sh's own IPO stage output -- "
                    "<run_dir>/LATEST/policy.pt.")
    p.add_argument("--dataset-path", required=True,
                    help="Resolved dataset JSON (build_personalization_dataset.py's own output) -- "
                         "read as plain JSON, no dpo_LRF import. Supplies the TEST prompts only.")
    p.add_argument("--train-path", required=True,
                    help="Raw train.json (user_id/prompt/chosen/rejected rows) -- the SAME file "
                         "PERSONALLLM_TRAIN_PATH points training at. Supplies the few-shot context "
                         "(NOT --dataset-path's own train_examples field, which lacks 'rejected').")
    p.add_argument("--output-path", required=True)
    p.add_argument("--model-name", default="Qwen/Qwen3-0.6B")
    p.add_argument("--num-shots", type=int, default=4,
                    help="MUST match the checkpoint's own training num_shots -- see "
                         "fspo_generic/dataset_data.py's own default.")
    p.add_argument("--include-dispreferred", action="store_true", default=True)
    p.add_argument("--max-prompt-length", type=int, default=512,
                    help="Cap on the (few-shot) prompt tokens, keep_start -- FSPO's own tokenize_batch_element truncation mode in training "
                         "(truncation_mode='keep_start' for every non-hh dataset). Same value as every other method's prompt cap "
                         "(SHARED_MAX_PROMPT_LENGTH). NOTE training only truncates when prompt+response exceeds max_length; here it is applied "
                         "unconditionally, since there is no response yet. 0 disables.")
    p.add_argument("--n-adapt-examples", type=int, default=None,
                    help="Caps each user's own --train-path pool to its first N rows before "
                         "--num-shots are taken from it -- same convention as eval/generate.py's "
                         "own icl branch / kadapter_dpo's own dim3 n-sweep (see chat: \"the unseen "
                         "jobs should be for 2,8,32\"). None (default) = uncapped, the SEEN "
                         "population's own behavior. If N < --num-shots, that user gets "
                         "min(--num-shots, N) shots instead of the full --num-shots (see chat: "
                         "\"would it be a problem to just use the two examples?!\" -> \"ok so use "
                         "the min thing\") -- a real, honest result under a tighter data budget, "
                         "not padded to look like more context than the user actually has.")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.dataset_path) as f:
        dataset = json.load(f)
    users = dataset["users"]

    with open(args.train_path) as f:
        train_rows = json.load(f)
    train_by_user = defaultdict(list)
    for r in train_rows:
        train_by_user[r["user_id"]].append(r)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"=== Loading base model ({args.model_name}) ===", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16)

    print(f"=== Loading IPO checkpoint from {args.archive} ===", flush=True)
    state_dict = torch.load(args.archive, map_location="cpu")
    step, metrics = state_dict["step_idx"], state_dict.get("metrics", {})
    print(f"  checkpoint step={step} metrics={metrics}", flush=True)
    if os.environ.get("FSPO_LORA", "0") == "1":
        # The checkpoint was written by a LoRA-wrapped policy (directpreferenceoptimization/train.py, FSPO_LORA=1): its keys are PeftModel keys (base_layer + lora_A/lora_B), so the
        # SAME LoraConfig (same env source as training: method_hyperparameters.py --method DPO's LORA_*) must wrap the model before load_state_dict(), then the adapter is merged for generation.
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(task_type="CAUSAL_LM", r=int(os.environ["LORA_R"]), lora_alpha=int(os.environ["LORA_ALPHA"]),
                                 lora_dropout=float(os.environ["LORA_DROPOUT"]), target_modules=os.environ["LORA_TARGET_MODULES"].split())
        model = get_peft_model(model, lora_config).to(torch.bfloat16)
        model.load_state_dict(state_dict["state"])
        model = model.merge_and_unload()
        print("  LoRA policy loaded and merged", flush=True)
    else:
        model.load_state_dict(state_dict["state"])
    model.eval()
    model.cuda()

    # checkpoint_hash sidecar -- see chat: "can you investigate why the
    # rewards are exactly the same as previously for PRLHF?" -> "yes for
    # any method!" -- same real incident/fix as Personalized_RLHF/
    # prlhf_personalllm/generate.py: a plain os.path.exists(args.output_
    # path) resume check (below) trusts cached generations regardless of
    # whether args.archive changed since they were written. FSPO's own
    # checkpoint is one file (a raw torch.save state dict), so this hashes
    # args.archive directly -- no PEFT-adapter-directory complication.
    checkpoint_hash = hashlib.sha256(open(args.archive, "rb").read()).hexdigest()[:16]

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    hash_path = args.output_path + ".checkpoint_hash"
    answers = {}
    if os.path.exists(args.output_path):
        stored_hash = open(hash_path).read().strip() if os.path.exists(hash_path) else None
        if stored_hash == checkpoint_hash:
            with open(args.output_path) as f:
                answers = json.load(f)
            n_done_already = sum(len(v) for v in answers.values())
            print(f"=== Resuming from {args.output_path} ({n_done_already} pairs already done, "
                  f"checkpoint_hash {checkpoint_hash} matches) ===", flush=True)
        else:
            print(f"=== {args.output_path} exists but its checkpoint_hash sidecar is "
                  f"missing/stale (stored={stored_hash}, current={checkpoint_hash}) -- the "
                  f"checkpoint changed since these generations were written. Ignoring cached "
                  f"generations, regenerating from scratch. ===", flush=True)
    with open(hash_path, "w") as f:
        f.write(checkpoint_hash)

    # Adaptation time: the construction of each user's own few-shot context
    # (still the unchanged first-num_shots-rows-of-the-pool slice -- see
    # chat: "and could you compute the 'adaptation time' as the
    # construction of the few shots?" -- the retrieval-based version of
    # this was proposed and explicitly turned down, this is just timing
    # what the script already does, not a mechanism change). Split into
    # its own pre-pass, timed as ONE block covering the whole population --
    # same "single total, not per-user/averaged" convention eval/
    # generate.py's own icl branch uses for its own adaptation_time_seconds.
    # Users with a POOL shorter than num_shots get min(num_shots,
    # len(pool)) shots instead -- NOT skipped -- see chat: "would it be a
    # problem to just use the two examples?!" -> "ok so use the min
    # thing". A real, mild input-distribution shift (training only ever
    # saw exactly-num_shots blocks), but the honest way to get an actual
    # N=2 data point instead of every user being skipped (which
    # --n-adapt-examples 2 would otherwise do, given num_shots=4) or a
    # downstream KeyError. n_short_pool counts these (informational, not
    # skipped); n_skipped_empty_pool counts the one real skip case left
    # (zero rows -- no context possible at all). Only ever fires with
    # --n-adapt-examples set below num_shots; the SEEN population
    # (n_adapt_examples=None, uncapped pool) essentially never hits either.
    n_short_pool = 0
    n_skipped_empty_pool = 0
    shots_by_user = {}
    _adapt_t0 = time.time()
    for uid, u in users.items():
        pool = train_by_user.get(uid, [])
        if args.n_adapt_examples is not None:
            pool = pool[:args.n_adapt_examples]
        if not pool:
            n_skipped_empty_pool += 1
            continue  # zero rows -- no context possible at all, a real skip.
        if len(pool) < args.num_shots:
            n_short_pool += 1
        shots = pool[:min(args.num_shots, len(pool))]  # no retrieval -- see module docstring.
        shots_by_user[uid] = (
            [s["prompt"] for s in shots], [s["chosen"] for s in shots], [s["rejected"] for s in shots],
        )
    adaptation_time_seconds = time.time() - _adapt_t0
    print(f"  built few-shot context for {len(shots_by_user)}/{len(users)} users "
          f"({adaptation_time_seconds:.3f}s)", flush=True)

    total_pairs = sum(len(u["prompts"]) for u in users.values())
    done = sum(len(v) for v in answers.values())
    n_truncated = 0
    _done0 = done
    t0 = time.time()
    with torch.no_grad():
        for uid, u in users.items():
            answers.setdefault(uid, {})
            shots_by_uid = shots_by_user.get(uid)
            if shots_by_uid is None:
                continue  # empty pool -- already counted in n_skipped_empty_pool above.
            fewshot_prompts, fewshot_preferred, fewshot_dispreferred = shots_by_uid

            for query in u["prompts"]:
                if query in answers[uid]:
                    continue
                # final_chosen/final_rejected are placeholders -- shots_to_text's
                # own chosen/rejected return values are TRAINING labels we
                # don't need here, only the "prompt" (few-shot text) return.
                prompt, _, _ = shots_to_text(
                    fewshot_prompts=fewshot_prompts,
                    fewshot_preferred=fewshot_preferred,
                    fewshot_dispreferred=fewshot_dispreferred,
                    final_question=query,
                    final_chosen="",
                    final_rejected="",
                    include_dispreferred=args.include_dispreferred,
                    add_persona=False,
                )
                input_ids = tokenizer(prompt, return_tensors="pt").input_ids
                if args.max_prompt_length and input_ids.shape[1] > args.max_prompt_length:
                    input_ids = input_ids[:, :args.max_prompt_length]   # keep_start, as in FSPO training
                    n_truncated += 1
                input_ids = input_ids.to("cuda")
                # do_sample=True, no temperature/top_p/top_k override -- see
                # chat: "I think I would use the Qwen recommended defaults
                # everywhere... For any method that is not ours, we just use
                # the default sampling parameters for the model" -> "no this
                # is not what I want! I want to just pick the default
                # parameters of the model" (NOT the model-card's "Best
                # Practices" numbers hardcoded as our own literal override --
                # the model's own actual shipped defaults). No `generation_
                # config=` object is built here on purpose: passing kwargs
                # straight to model.generate() lets it fall through to
                # self.generation_config (Qwen3-0.6B's own generation_config.
                # json, confirmed: do_sample=True, temperature=0.6, top_p=
                # 0.95, top_k=20) for every field this call doesn't set --
                # building a fresh GenerationConfig(...) object instead would
                # silently replace those with the GenerationConfig CLASS's
                # own defaults (temperature=1.0, top_p=1.0, top_k=50),
                # nothing to do with the model at all. load_state_dict()
                # above only overwrites weights, never touches
                # model.generation_config, so this is genuinely the same
                # object Qwen shipped.
                output = model.generate(
                    input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                    max_new_tokens=MAX_NEW_TOKENS, do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                )
                new_tokens = output[0][input_ids.shape[1]:]
                decoded = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                answers[uid][query] = _strip_response_header(decoded)

                done += 1
                with open(args.output_path, "w") as f:
                    json.dump(answers, f, indent=2)
                if done % 20 == 0 or done == total_pairs:
                    print(f"  generated {done}/{total_pairs} ({time.time() - t0:.0f}s elapsed)", flush=True)

    if n_skipped_empty_pool:
        print(f"  {n_skipped_empty_pool}/{len(users)} users skipped (zero train_examples "
              f"available)", flush=True)
    if n_short_pool:
        print(f"  {n_short_pool}/{len(users)} users given fewer than {args.num_shots} shots "
              f"(pool shorter than --num-shots, not skipped -- see chat: \"use the min thing\")",
              flush=True)

    print(f"=== {n_truncated} prompts truncated to {args.max_prompt_length} tokens (keep_start) ===", flush=True)
    print(f"wrote {args.output_path}", flush=True)

    # Timing sidecar, same convention as dpo_LRF/eval/generate.py's own
    # _write_timing and prlhf_personalllm/generate.py's own equivalent --
    # downstream_eval.py already looks for generation_timing_alpha{alpha}.json
    # next to --personalized-generations-path, and merges WHATEVER keys are
    # in it into the registry write unchanged -- adaptation_time_seconds
    # added here (see chat: "could you compute the 'adaptation time' as
    # the construction of the few shots?") needs no downstream_eval.py
    # changes at all, same as icl's own adaptation_time_seconds already
    # works.
    with open(os.path.join(os.path.dirname(args.output_path),
                            f"generation_timing_alpha{dataset['metadata']['alpha']}.json"), "w") as f:
        import socket as _socket
        # timing_scope v2 (dpo_LRF/timing_scope.py): model/tokenizer/checkpoint loading excluded (all before t0 / _adapt_t0); adaptation =
        # building each user's few-shot context; inference = this invocation's generation loop (a resumed run only times what it generated).
        json.dump({"timing_scope": "v2_no_model_loading", "inference_time_seconds": time.time() - t0,
                    "adaptation_time_seconds": adaptation_time_seconds,
                    "hardware": f"{torch.cuda.get_device_name(0)} x{torch.cuda.device_count()} @ {_socket.gethostname()}",
                    "n_generations": total_pairs, "n_generations_this_run": done - _done0}, f)

    print("[INFO] Script execution completed successfully.", flush=True)


if __name__ == "__main__":
    main()
