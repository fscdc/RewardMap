export LMUData="path/to/LMUData"
export HF_TORCH_DTYPE=bfloat16

torchrun --nproc-per-node=8 path/to/VLMEvalKit/run.py \
    --data SEEDBench2_Plus SpatialEval VStarBench HRBench4K ChartQA_TEST MMStar \
    --model your_model