export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

set -x

export VLLM_ATTENTION_BACKEND=XFORMERS

MODEL_PATH=path/to/sft_trained_model  # replace it with your local file path

RUN_NAME=$(basename "$0" .sh)

python3 -m verl.trainer.main \
    config=scripts/train_sft_rl.yaml \
    data.train_files=data/ReasonMap-Train \
    data.val_files=None \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.kl_loss_coef=1.0e-2 \
    worker.actor.optim.lr=1.0e-6 \
    worker.actor.micro_batch_size_per_device_for_update=1 \
    worker.actor.micro_batch_size_per_device_for_experience=1 \
    worker.rollout.tensor_parallel_size=4 \
    worker.rollout.gpu_memory_utilization=0.6 \
    worker.rollout.enable_chunked_prefill=false \
    worker.rollout.n=4 \
    worker.reward.compute_score=reason_map \
    trainer.experiment_name=${RUN_NAME} \
    trainer.n_gpus_per_node=8 \
    trainer.total_episodes=1 \
    trainer.save_checkpoint_path=reason_map/${RUN_NAME}_output