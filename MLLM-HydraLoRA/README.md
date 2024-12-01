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
bash HydraLoRA/fine-tuning.sh
```

**Details**
```
CUDA_VISIBLE_DEVICES=1 python llava/train/train_mem.py \
    --use_lora True --lora_rank 32 --lora_alpha 64 --mm_projector_lr 2e-5 \
    --llm_moe True --dense_moe True --lora_modules q_proj,k_proj,v_proj,o_proj --llm_moe_num_experts 3 --moe_balance_w 0.05 \
    --model_name_or_path ./checkpoints/llava-v1.5-7b \
    --version v1 \
    --freeze_backbone True \
    --data_path ./playground/data/llava_selected_60k.json \
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
    --output_dir ./checkpoints/llava-v1.5-7b-mohle-60k \
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

## ⭐ Citation

If you find our work helpful, please consider citing our paper:

```
@inproceedings{
tian2024hydralora,
title={HydraLo{RA}: An Asymmetric Lo{RA} Architecture for Efficient Fine-Tuning},
author={Chunlin Tian and Zhan Shi and Zhijiang Guo and Li Li and Cheng-zhong Xu},
booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
year={2024},
url={https://openreview.net/forum?id=qEpi8uWX3N}
}
```

## ❤️ References

The code refers to the repo [LLaVA-MoLE](https://github.com/forwchen/LLaVA-MoLE), [LLaVA](https://github.com/haotian-liu/LLaVA). Thanks to [Xuyang](https://github.com/coder23j) for organizing the relevant code.