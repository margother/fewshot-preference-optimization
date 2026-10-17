import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.backends
import torch.nn as nn
import transformers
from utils import get_local_dir, get_local_run_dir, disable_dropout, init_distributed, get_open_port
import os
import hydra
import torch.multiprocessing as mp
from omegaconf import OmegaConf, DictConfig
import trainers
import wandb
import json
import socket
from typing import Optional, Set
import resource


OmegaConf.register_new_resolver("get_local_run_dir", lambda exp_name, local_dirs: get_local_run_dir(exp_name, local_dirs))


def worker_main(rank: int, world_size: int, config: DictConfig, policy: nn.Module, reference_model: Optional[nn.Module] = None):
    """Main function for each worker process (may be only 1 for BasicTrainer/TensorParallelTrainer)."""
    if 'FSDP' in config.trainer:
        init_distributed(rank, world_size, port=config.fsdp_port)
    
    if config.debug:
        wandb.init = lambda *args, **kwargs: None
        wandb.log = lambda *args, **kwargs: None

    if rank == 0 and config.wandb.enabled:
        os.environ['WANDB_CACHE_DIR'] = get_local_dir(config.local_dirs)
        wandb.init(
            entity=config.wandb.entity,
            project=config.wandb.project,
            config=OmegaConf.to_container(config),
            dir=get_local_dir(config.local_dirs),
            name=config.exp_name,
        )

    TrainerClass = getattr(trainers, config.trainer)
    print(f'Creating trainer on process {rank} with world size {world_size}')
    trainer = TrainerClass(policy, config, config.seed, config.local_run_dir, reference_model=reference_model, rank=rank, world_size=world_size)

    trainer.train()
    trainer.save()


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig):
    """Main entry point for training. Validates config, creates/initializes model(s), and kicks off worker process(es)."""

    # Resolve hydra references, e.g. so we don't re-compute the run directory
    OmegaConf.resolve(config)

    missing_keys: Set[str] = OmegaConf.missing_keys(config)
    if missing_keys:
        raise ValueError(f"Got missing keys in config:\n{missing_keys}")

    if config.eval_every % config.batch_size != 0:
        print('WARNING: eval_every must be divisible by batch_size')
        print('Setting eval_every to', config.eval_every - config.eval_every % config.batch_size)
        config.eval_every = config.eval_every - config.eval_every % config.batch_size

    if 'FSDP' in config.trainer and config.fsdp_port is None:
        free_port = get_open_port()
        print('no FSDP port specified; using open port for FSDP:', free_port)
        config.fsdp_port = free_port

    print(OmegaConf.to_yaml(config))

    config_path = os.path.join(config.local_run_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        OmegaConf.save(config, f)

    print('=' * 80)
    print(f'Writing to {socket.gethostname()}:{config.local_run_dir}')
    print('=' * 80)
 
    os.environ['XDG_CACHE_HOME'] = get_local_dir(config.local_dirs)
    print('building policy')
    model_kwargs = {'device_map': 'balanced'} if config.trainer == 'BasicTrainer' else {    "attn_implementation": "flash_attention_2"
}
    policy_dtype = getattr(torch, config.model.policy_dtype)
    policy = transformers.AutoModelForCausalLM.from_pretrained(
        config.model.name_or_path, cache_dir=get_local_dir(config.local_dirs), low_cpu_mem_usage=True, torch_dtype=policy_dtype, **model_kwargs)
    disable_dropout(policy)
    # Single-GPU path (trainer=BasicTrainer): FSDP is not used, so its activation checkpointing (FSDPTrainer.__init__'s apply_activation_checkpointing) does not exist either -- use HF's own
    # gradient checkpointing instead, non-reentrant (the sequences are up to 6144 tokens: without it the activations of 28 layers do not fit next to the two model copies). Done BEFORE the
    # LoRA wrap so PEFT inherits it. FSDPTrainer runs are untouched.
    if config.trainer == 'BasicTrainer' and config.activation_checkpointing:
        policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        print('gradient checkpointing enabled (HF, non-reentrant) for the BasicTrainer policy')

    # PERSONALLLM_LORA opt-in -- see chat: "does IPO use LoRa?!" -> "is it
    # trivial to change the fine-tuning to Lora in their code?" -> "I think
    # this is fair to the other methods... run a new sanity check with
    # fewshot=8 but a lora adaptor with the same parameters we've been
    # using". Full fine-tuning (self.optimizer over self.policy.parameters()
    # -- ALL of them, trainers.py's own real mechanism, confirmed directly)
    # is what caused every prior IPO-stage OOM here, none of which were
    # about LoRA at all -- this makes FSPO's own memory profile match the
    # other (already LoRA-based) methods instead of full-fine-tuning a
    # whole 3B model + full RMSprop state for it. Same r/alpha/dropout/
    # target_modules DPO/D-DPO/DPO-LRF's own canonical LoRA config already
    # uses (method_hyperparameters.py), not a fresh choice. reference_model
    # (below) deliberately stays a plain full frozen copy, NOT also LoRA-
    # wrapped -- no gradient/optimizer memory to save there since it's
    # already frozen, and it keeps this change minimal/localized rather
    # than restructuring trainers.py's own concatenated_forward to a
    # reference-free (same-weights-adapter-disabled) scheme. Env-var gated,
    # default off -- qwen's own existing full-fine-tuning pipeline stays
    # byte-identical unless a launcher explicitly opts in (same pattern as
    # PERSONALLLM_NUM_SHOTS).
    if os.environ.get("PERSONALLLM_LORA", "0") == "1":
        from peft import LoraConfig, get_peft_model
        # LoRA hyperparameters come from the SAME source of truth DPO / DPO-LRF / D-DPO use -- dpo_LRF/method_hyperparameters.py (LORA_R / LORA_ALPHA / LORA_DROPOUT /
        # LORA_TARGET_MODULES, exported by `method_hyperparameters.py --method DPO`), so "FSPO uses the same LoRA as DPO" is true by construction, not by a copy of the numbers
        # here (see chat: "I want it to run with the same ranks etc. that we use for DPO"). Fallback = the current canonical values, with a loud warning, only so an older launcher
        # that does not export them keeps working.
        if all(k in os.environ for k in ("LORA_R", "LORA_ALPHA", "LORA_DROPOUT", "LORA_TARGET_MODULES")):
            _lora = dict(r=int(os.environ["LORA_R"]), lora_alpha=int(os.environ["LORA_ALPHA"]), lora_dropout=float(os.environ["LORA_DROPOUT"]),
                         target_modules=os.environ["LORA_TARGET_MODULES"].split())
        else:
            print("WARNING: LORA_R/LORA_ALPHA/LORA_DROPOUT/LORA_TARGET_MODULES not exported -- using the hardcoded canonical DPO values (r=8, alpha=16, dropout=0.05, q_proj v_proj)")
            _lora = dict(r=8, lora_alpha=16, lora_dropout=0.05, target_modules=["q_proj", "v_proj"])
        print(f"LoRA config: {_lora}")
        lora_config = LoraConfig(task_type="CAUSAL_LM", **_lora)
        policy = get_peft_model(policy, lora_config)
        # get_peft_model() creates the new lora_A/lora_B adapter matrices
        # in float32 by default, regardless of the base model's own dtype
        # -- real bug found in job 330127 (sanity check): FSDP's own
        # FlatParamHandle refuses to shard a module whose parameters have
        # mixed dtypes ("Must flatten tensors with uniform dtype but got
        # torch.bfloat16 and torch.float32"), which every base param here
        # already is (policy_dtype above). Cast the whole (now LoRA-
        # wrapped) model back to policy_dtype -- a no-op on the already-
        # bfloat16 base weights, fixes only the new adapter params.
        policy = policy.to(policy_dtype)
        policy.print_trainable_parameters()

    if config.loss.name in {'dpo', 'ipo'}:
        print('building reference model')
        reference_model_dtype = getattr(torch, config.model.reference_dtype)
        reference_model = transformers.AutoModelForCausalLM.from_pretrained(
            config.model.name_or_path, cache_dir=get_local_dir(config.local_dirs), low_cpu_mem_usage=True, torch_dtype=reference_model_dtype, **model_kwargs)
        disable_dropout(reference_model)
        # MUST also be LoRA-wrapped, same config as policy above, when
        # PERSONALLLM_LORA=1 -- NOT for its own training (reference_model
        # is never in self.optimizer's own parameter list, trainers.py's
        # own real mechanism, confirmed directly -- it's pure eval/no_grad
        # forward passes), but because config.model.archive's own
        # reference_model.load_state_dict(state_dict['state']) a few lines
        # below loads the SAME state dict the (now PEFT-wrapped) policy
        # saved -- a plain, unwrapped model would key-mismatch against a
        # PeftModel's own state_dict (renamed/prefixed base keys + real new
        # lora_A/lora_B tensors that don't exist on a plain model at all).
        # This also correctly represents WHAT the IPO stage's reference
        # actually needs to be: the SFT-tuned policy (base + its trained
        # adapter), not the raw pretrained model with no adapter at all.
        if os.environ.get("PERSONALLLM_LORA", "0") == "1":
            reference_model = get_peft_model(reference_model, lora_config)
            # Same float32-adapter-vs-bfloat16-base dtype fix as policy above.
            reference_model = reference_model.to(reference_model_dtype)
    else:
        reference_model = None

    if config.model.archive is not None:
        state_dict = torch.load(config.model.archive, map_location='cpu')
        step, metrics = state_dict['step_idx'], state_dict['metrics']
        print(f'loading pre-trained weights at step {step} from {config.model.archive} with metrics {json.dumps(metrics, indent=2)}')
        policy.load_state_dict(state_dict['state'])
        if config.loss.name in {'dpo', 'ipo'}:
            reference_model.load_state_dict(state_dict['state'])
        print('loaded pre-trained weights')
    
    if 'FSDP' in config.trainer:
        world_size = torch.cuda.device_count()
        print('starting', world_size, 'processes for FSDP training')
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
        print(f'setting RLIMIT_NOFILE soft limit to {hard} from {soft}')
        mp.spawn(worker_main, nprocs=world_size, args=(world_size, config, policy, reference_model), join=True)
    else:
        print('starting single-process worker')
        worker_main(0, 1, config, policy, reference_model)


if __name__ == '__main__':
    main()