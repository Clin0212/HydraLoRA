# MLLM_HydraLoRA: An Asymmetric LoRA Architecture for Multimodal LLMs (Llava)


## 🛠️ Install

1. **Environment**

```
conda create -n MLLM_HydraLoRA python=3.10 -y
conda activate MLLM_HydraLoRA
# get LLaVa base code
git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA
git checkout c121f04
pip install -e .
# install main packages
pip install requirements.txt
```

2. **Patch Llava**

```
./moe_patch/patch.sh
cp moe.py llava/model/
cp builder.py llava/model/
cp llava_llama.py llava/model/language_model/
cp train.py llava/train/
```

## 🕹️ Quickstart (Fine-tuning)

**MLLM_HydraLoRA Training**

```
bash HydraLoRA/MLLM_HydraLoRA.sh
```

**Details**
```
CUDA_VISIBLE_DEVICES=0 python llava/train/train_mem.py \
    --use_lora True --lora_rank 32 --lora_alpha 64 --mm_projector_lr 2e-5 \
    --llm_moe True --dense_moe True --lora_modules q_proj,k_proj,v_proj,o_proj --llm_moe_num_experts 3 --moe_balance_w 0.05 \
    --model_name_or_path ./checkpoints/llava-v1.5-7b \
    --version v1 \
    --freeze_backbone True \
    --data_path ./playground/data/llava_v1_5_mix665k.json \
    --image_folder ./playground/data \
    --vision_tower ./checkpoints/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter ./checkpoints/llava-v1.5-7b/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio resize \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/llava-v1.5-7b-lora-mohle \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 2 \
    --lazy_preprocess True

```
## 🌋 Evaluation(TextVQA for example)

**Prepare for evaluation**  
check [scripts](https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md#scripts) and [data](https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md#textvqa)

**Gen answers**
```
python -m llava.eval.model_vqa_loader \
    --model-base checkpoints/llava-v1.5-7b \
    --model-path checkpoints/llava-v1.5-7b-lora-mohle \
    --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder ./playground/data/eval/textvqa/train_images \
    --answers-file ./eval-output/textvqa/llava-mohle-7b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

```
**Evaluate**
```
python -m llava.eval.eval_textvqa \
    --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file ./eval-output/textvqa/llava-mohle-7b.jsonl

```
## ⭐ Citation

If you find our work helpful, please consider citing our paper:

```
@inproceedings{
tian2024hydralora,
title={HydraLoRA: An Asymmetric LoRA Architecture for Efficient Fine-Tuning},
author={Chunlin Tian and Zhan Shi and Zhijiang Guo and Li Li and Cheng-zhong Xu},
booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
year={2024},
url={https://openreview.net/forum?id=qEpi8uWX3N}
}
```

## ❤️ References

The code refers to the repo [LLaVA-MoLE](https://github.com/forwchen/LLaVA-MoLE), [LLaVA](https://github.com/haotian-liu/LLaVA). Thanks to [Xuyang](https://github.com/coder23j) for organizing the relevant code.
