"""
Test-set preference accuracy for a trained FSPO (SFT+IPO) checkpoint --
FSPO analogue of Personalized_RLHF/prlhf_personalllm/compute_test_accuracy.py
(see that file's own module docstring for the shared rationale: neither
method registered a test-set accuracy before this session, see chat: "can
we rather use their code?").

Reuses BasicTrainer's OWN get_batch_metrics (trainers.py) unmodified for
the actual forward-pass/reward/accuracy computation -- constructed directly
(NOT through train.py's own @hydra.main entry point, which is CLI-only) so
this can be called as a plain post-training step. Single GPU (rank=0,
world_size=1) is enough here: eval-only, no FSDP sharding needed (unlike
the real 2-GPU IPO training this checkpoint came from).

Reference model: NOT the raw base model -- train.py's own IPO stage loads
BOTH policy and reference_model from the SFT archive when model.archive is
set (see train.py: "if config.loss.name in {'dpo', 'ipo'}: reference_model.
load_state_dict(state_dict['state'])"). This script reproduces that exactly
via --sft-archive, so the "reference" here matches what the checkpoint was
ACTUALLY trained against, not some other baseline.

Test split: get_fewshot_dataset() is now split-aware (see fspo_generic/
dataset_data.py's own comment) -- split="test" reads PERSONALLLM_
TEST_PATH and windows it with the SAME num_shots/reshuffle mechanism as
training, just applied to the held-out test.json instead of train.json
(both few-shot context AND target question/chosen/rejected come from the
test split -- mirrors get_elix/get_review/get_roleplay's own real train/
test symmetry, not a train-context-into-test-target mix). PERSONALLLM_
TRAIN_PATH is pointed at a tiny 2-row dummy file (BasicTrainer.__init__
unconditionally builds a train_iterator too, even though .train() is
never called here -- a real dummy avoids re-running the ~36k-example
reshuffle-augmentation on the full train.json for nothing, same fix as
the P-RLHF script's own dummy_train_dataset).

Formula (trainers.py's own preference_loss, ipo=True branch -- SAME reward
definition DPO uses, only the LOSS differs):
    chosen_reward   = beta * (policy_chosen_logp   - reference_chosen_logp)
    rejected_reward = beta * (policy_rejected_logp - reference_rejected_logp)
    accuracy = mean(chosen_reward > rejected_reward)

Usage:
    python compute_test_accuracy.py \\
        --archive /path/to/personalllm_seen_ipo_TIMESTAMP/LATEST/policy.pt \\
        --sft-archive /path/to/personalllm_seen_sft_TIMESTAMP/LATEST/policy.pt \\
        --test-path /path/to/test.json \\
        --output-path /path/to/test_accuracy.json
"""
import argparse
import json
import os
import sys
import tempfile
import time

import torch
import transformers
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "directpreferenceoptimization"))
from trainers import BasicTrainer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--archive", required=True, help="train_seen.sh's own IPO stage output -- "
                    "<run_dir>/LATEST/policy.pt.")
    p.add_argument("--sft-archive", required=True, help="train_seen.sh's own SFT stage output -- "
                    "the SAME archive the IPO stage's own reference model was loaded from.")
    p.add_argument("--test-path", required=True,
                    help="Raw test.json (user_id/prompt/chosen/rejected rows) -- the SAME held-out "
                         "split dim1/dim3/P-RLHF's own accuracy blocks use.")
    p.add_argument("--output-path", required=True)
    p.add_argument("--model-name", default="Qwen/Qwen3-0.6B")
    p.add_argument("--num-shots", type=int, default=4,
                    help="MUST match the checkpoint's own training num_shots -- see "
                         "fspo_generic/dataset_data.py's own default.")
    p.add_argument("--beta", type=float, default=0.01,
                    help="MUST match the checkpoint's own training loss.beta -- doesn't affect "
                         "accuracy itself (beta-invariant sign comparison, see module docstring), "
                         "kept correct for clarity, same as the P-RLHF script's own --beta.")
    p.add_argument("--max-prompt-length", type=int, default=5632)
    p.add_argument("--max-length", type=int, default=6144)
    p.add_argument("--n-eval-examples", type=int, default=256,
                    help="Caps the windowed test set the same way train.py's own default "
                         "n_eval_examples (256, config.yaml) does -- NOT the full test.json, see "
                         "module docstring on tractability.")
    p.add_argument("--eval-batch-size", type=int, default=2)
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()

    os.environ["PERSONALLLM_TEST_PATH"] = args.test_path
    # Tiny dummy train file -- see module docstring. Needs >= num_shots+2 rows for >= 1 user so
    # get_personalllm()'s own windowing doesn't produce an empty train_iterator (harmless either
    # way here, but avoids a confusing "0 batches" print).
    dummy_rows = [
        {"user_id": "dummy_user", "prompt": f"dummy prompt {i}", "chosen": "a", "rejected": "b"}
        for i in range(args.num_shots + 2)
    ]
    dummy_train_path = os.path.join(tempfile.gettempdir(), f"fspo_dummy_train_{os.getpid()}.json")
    with open(dummy_train_path, "w") as f:
        json.dump(dummy_rows, f)
    os.environ["PERSONALLLM_TRAIN_PATH"] = dummy_train_path

    config = OmegaConf.create({
        "datasets": ["personalllm"],
        "max_length": args.max_length,
        "max_prompt_length": args.max_prompt_length,
        "batch_size": 2,          # throwaway train_iterator only -- never used, see module docstring.
        "eval_batch_size": args.eval_batch_size,
        "n_epochs": None,
        "n_examples": 2,          # throwaway train_iterator cap, matches the dummy file's own tiny size.
        "n_eval_examples": args.n_eval_examples,
        "trainer": "BasicTrainer",
        "loss": {"name": "ipo", "beta": args.beta},
        "model": {"name_or_path": args.model_name, "tokenizer_name_or_path": None},
        "local_dirs": [tempfile.gettempdir()],
    })

    print(f"=== Loading policy (IPO checkpoint: {args.archive}) ===", flush=True)
    policy = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_name, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16)
    policy_state = torch.load(args.archive, map_location="cpu")
    print(f"  IPO checkpoint step={policy_state['step_idx']} metrics={policy_state.get('metrics', {})}", flush=True)
    policy.load_state_dict(policy_state["state"])

    print(f"=== Loading reference (SFT checkpoint: {args.sft_archive}) -- matches what IPO's own "
          f"training actually used, NOT the raw base model ===", flush=True)
    reference_model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_name, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16)
    sft_state = torch.load(args.sft_archive, map_location="cpu")
    reference_model.load_state_dict(sft_state["state"])

    policy.cuda()
    reference_model.cuda()
    reference_model.eval()

    print("=== Building BasicTrainer (rank=0, world_size=1 -- eval-only, no FSDP needed) ===", flush=True)
    trainer = BasicTrainer(policy, config, seed=0, run_dir=tempfile.gettempdir(),
                            reference_model=reference_model, rank=0, world_size=1)

    print(f"=== Running {len(trainer.eval_batches)} eval batches through BasicTrainer's own "
          f"get_batch_metrics (train=False) ===", flush=True)
    trainer.policy.eval()
    all_accuracies = []
    with torch.no_grad():
        for i, batch in enumerate(trainer.eval_batches):
            _, metrics = trainer.get_batch_metrics(batch, config.loss, train=False)
            all_accuracies.extend(metrics["rewards_eval/accuracies"])
            if (i + 1) % 20 == 0:
                print(f"  {i + 1}/{len(trainer.eval_batches)} eval batches done "
                      f"({time.time() - t0:.0f}s elapsed)", flush=True)

    population_accuracy = sum(all_accuracies) / len(all_accuracies)
    n_pairs = len(all_accuracies)
    print(f"=== population_accuracy={population_accuracy:.4f} over {n_pairs} windowed examples ===",
          flush=True)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump({
            "accuracy": population_accuracy,
            "n_pairs": n_pairs,
            "n_eval_examples": args.n_eval_examples,
            "archive": args.archive,
            "sft_archive": args.sft_archive,
        }, f, indent=2)
    print(f"wrote {args.output_path}", flush=True)

    os.remove(dummy_train_path)
    print("[INFO] Script execution completed successfully.", flush=True)


if __name__ == "__main__":
    main()
