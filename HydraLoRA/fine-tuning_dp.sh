export CUDA_HOME=/usr/local/cuda-11.8
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64
export PATH=${CUDA_HOME}/bin:${PATH}

# User-defined parameters
lr=0.0002
lora_rank=4
lora_alpha=32
lora_trainable="gate_proj,down_proj,up_proj"
lora_dropout=0.05                 
pretrained_model= path_to_llama_model
tokenizer_path= path_to_llama_model
dataset_dir= data/example_data
validation_file= data/example_data/test.json
per_device_train_batch_size= 1
per_device_eval_batch_size= 1
gradient_accumulation_steps= 8
max_seq_length= 1024
output_dir= path_to_output
deepspeed_config_file= ds_zero3_offload.json
exp_name=lora_model

lora_b_nums= 4  # Developer-specific, k-means, or DBSCAN et al.





CUDA_VISIBLE_DEVICES=0,1,2,3 \
CUDA_LAUNCH_BLOCKING=1 \
torchrun --nnodes 1 --nproc_per_node 4 --node_rank 0 --master_port 29502 \
    fine-tuning.py \
    --deepspeed ${deepspeed_config_file} \
    --model_name_or_path ${pretrained_model} \
    --tokenizer_name_or_path ${tokenizer_path} \
    --dataset_dir ${dataset_dir} \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size ${per_device_eval_batch_size} \
    --do_train \
    --do_eval \
    --seed 41 \
    --bf16 \
    --num_train_epochs 1 \
    --lr_scheduler_type cosine \
    --learning_rate ${lr} \
    --warmup_ratio 0.03 \
    --weight_decay 0 \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --save_total_limit 5 \
    --evaluation_strategy steps \
    --eval_steps 5000 \
    --save_steps 5000 \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --preprocessing_num_workers 8 \
    --max_seq_length ${max_seq_length} \
    --output_dir ${output_dir}/${exp_name} \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --lora_rank ${lora_rank} \
    --lora_alpha ${lora_alpha} \
    --lora_nums ${lora_b_nums} \
    --trainable ${lora_trainable} \
    --lora_dropout ${lora_dropout} \
    --torch_dtype bfloat16 \
    --validation_file ${validation_file} \
    --load_in_kbits 16 \
    --ddp_find_unused_parameters False \
    --overwrite_output_dir \
  
