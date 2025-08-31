import importlib.util
import json
import os
from pprint import pprint

import hydra
import numpy as np
import ray
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager, Role
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.fs import copy_to_local

DEFAULT_REWARD_THRESHOLD = 1.0


class DifficultyFilter(RayPPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reward_threshold = DEFAULT_REWARD_THRESHOLD
        self.reward_threshold_dict = None
        try:
            threshold = self.config.difficulty_filter.reward_threshold
            if isinstance(threshold, float):
                self.reward_threshold = threshold
            elif isinstance(threshold, str):
                with open(threshold, "r") as f:
                    self.reward_threshold_dict = json.load(f)
        except AttributeError:
            print(
                f"No reward threshold found in config. Using default value: {self.reward_threshold}"
            )

    def sample_and_score(self, dataloader: DataLoader) -> dict[tuple[str, int], dict]:
        reward_tensor_lst = []
        data_source_lst = []

        # lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []
        # pass rate
        results = {}

        for batch_idx, test_data in enumerate(dataloader):
            test_batch = DataProto.from_single_dict(test_data)

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n,
                interleave=True,
            )

            # we only do validation on rule-based rm
            if (
                self.config.reward_model.enable
                and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model"
            ):
                return {}

            # store original inputs
            input_ids = test_batch.batch["input_ids"]
            input_texts = [
                self.tokenizer.decode(ids, skip_special_tokens=True)
                for ids in input_ids
            ]
            sample_inputs.extend(input_texts)

            if "multi_modal_inputs" in test_batch.non_tensor_batch.keys():
                test_gen_batch = test_batch.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"],
                    non_tensor_batch_keys=[
                        "raw_prompt_ids",
                        "multi_modal_data",
                        "multi_modal_inputs",
                    ],
                )
            else:
                test_gen_batch = test_batch.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"],
                    non_tensor_batch_keys=["raw_prompt_ids"],
                )

            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
            }
            # print(f'test_gen_batch meta info: {test_gen_batch.meta_info}')

            # pad to be divisible by dp_size
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(
                test_gen_batch, self.actor_rollout_wg.world_size
            )
            test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(
                test_gen_batch_padded
            )

            # unpad
            test_output_gen_batch = unpad_dataproto(
                test_output_gen_batch_padded, pad_size=pad_size
            )
            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [
                self.tokenizer.decode(ids, skip_special_tokens=True)
                for ids in output_ids
            ]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            # shape: (batch_size, seq_len)
            reward_tensor = self.val_reward_fn(test_batch)

            # need to aggregate over repeats
            rollout_reward_tensor = reward_tensor.sum(-1)
            num_samples_per_example = self.config.difficulty_filter.num_samples_per_example
            data_sources = test_batch.non_tensor_batch.get(
                "data_source", ["unknown"] * reward_tensor.shape[0]
            )
            example_indices = test_batch.non_tensor_batch.get(
                "index", [-1] * reward_tensor.shape[0]
            )
            for i in range(0, rollout_reward_tensor.shape[0], num_samples_per_example):
                example_rewards = rollout_reward_tensor[i : i + num_samples_per_example]
                result_key = (data_sources[i], example_indices[i])
                pass_rate = (example_rewards >= self.reward_threshold).float().mean().item()
                results[result_key] = {
                    "rewards": example_rewards.tolist(),
                    "pass_rate": pass_rate,
                    "batch_idx": batch_idx,
                    "example_idx": i // num_samples_per_example,
                }

            # store scores
            scores = rollout_reward_tensor.cpu().tolist()
            sample_scores.extend(scores)
            reward_tensor_lst.append(reward_tensor)
            data_source_lst.append(data_sources)

        self._maybe_log_val_generations(
            inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores
        )
        reward_tensor = (
            torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()
        )  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)

        # evaluate test_score based on data source
        data_source_reward = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())

        return results


def get_custom_reward_fn(config):
    reward_fn_config = config.get("custom_reward_function") or {}
    file_path = reward_fn_config.get("path")
    if not file_path:
        return None

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Reward function file '{file_path}' not found.")

    spec = importlib.util.spec_from_file_location("custom_module", file_path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise RuntimeError(f"Error loading module from '{file_path}': {e}")

    function_name = reward_fn_config.get("name")

    if not hasattr(module, function_name):
        raise AttributeError(
            f"Reward function '{function_name}' not found in '{file_path}'."
        )

    print(f"using customized reward function '{function_name}' from '{file_path}'")

    return getattr(module, function_name)


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
def main_task(config):
    pprint(
        OmegaConf.to_container(config, resolve=True)
    )  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # download the checkpoint from hdfs
    local_path = copy_to_local(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    tokenizer = hf_tokenizer(local_path)
    processor = hf_processor(
        local_path, use_fast=True
    )  # used for multimodal LLM, could be none

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == "fsdp":
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.single_controller.ray import RayWorkerGroup
        from verl.workers.fsdp_workers import ActorRolloutRefWorker

        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == "megatron":
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        from verl.workers.megatron_workers import ActorRolloutRefWorker

        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    role_worker_mapping = {Role.ActorRollout: ray.remote(ActorRolloutRefWorker)}

    global_pool_id = "global_pool"
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        # Role.Critic: global_pool_id,
        # Role.RefPolicy: global_pool_id,
    }

    # we should adopt a multi-source reward function here
    # - for rule-based rm, we directly call a reward score
    # - for model-based rm, we call a model
    # - for code related prompt, we send to a sandbox if there are test cases
    # - finally, we combine all the rewards together
    # - The reward type depends on the tag of the data
    if config.reward_model.enable:
        if config.reward_model.strategy == "fsdp":
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == "megatron":
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    reward_manager_name = config.reward_model.get("reward_manager", "naive")
    if reward_manager_name == "naive":
        from verl.workers.reward_manager import NaiveRewardManager

        reward_manager_cls = NaiveRewardManager
    elif reward_manager_name == "prime":
        from verl.workers.reward_manager import PrimeRewardManager

        reward_manager_cls = PrimeRewardManager
    else:
        raise NotImplementedError

    compute_score = get_custom_reward_fn(config)
    reward_fn = reward_manager_cls(
        tokenizer=tokenizer, num_examine=0, compute_score=compute_score
    )

    # note that we always use function-based RM for validation
    val_reward_fn = reward_manager_cls(
        tokenizer=tokenizer, num_examine=1, compute_score=compute_score
    )

    resource_pool_manager = ResourcePoolManager(
        resource_pool_spec=resource_pool_spec, mapping=mapping
    )

    filter_pipeline = DifficultyFilter(
        config=config,
        tokenizer=tokenizer,
        processor=processor,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=ray_worker_group_cls,
        reward_fn=reward_fn,
        val_reward_fn=val_reward_fn,
    )
    filter_pipeline.init_workers()
    scores = filter_pipeline.sample_and_score(
        dataloader=filter_pipeline.train_dataloader
    )
    if config.difficulty_filter.output_file:
        with open(config.difficulty_filter.output_file, "w") as f:
            json.dump(scores, f)
            print(f"Results saved to {config.difficulty_filter.output_file}")
    return scores


def run_sample_and_score(config) -> None:
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(
            runtime_env={
                "env_vars": {
                    "TOKENIZERS_PARALLELISM": "true",
                    "NCCL_DEBUG": "WARN",
                    "VLLM_LOGGING_LEVEL": "WARN",
                }
            }
        )
    ray.get(main_task.remote(config))


@hydra.main(config_path="config", config_name="diff_filter", version_base=None)
def main(config):
    run_sample_and_score(config)


if __name__ == "__main__":
    main()
