save_dir="eval_results/"
data_dir="data/"
model_path="/vepfs/hongcheng/moe/modelscope_cache/deepseek-ai/DeepSeek-V2-Lite"
#peft_model="."
#prune_result="."
#peft_model="/vepfs/hongcheng/moe/MoE_unsupervised_pruning/result/train_2500_valid_250/pruning_finetuning/sample_500_cluster_12/in_class_prune_0.2"
#prune_result="/vepfs/hongcheng/moe/MoE_unsupervised_pruning/pruned_result/sample_500_cluster_12/in_class_prune_0.2"
peft_model="."
prune_result="/vepfs/hongcheng/moe/MoE_unsupervised_pruning/pruned_result/sample_500_cluster_12/in_class_prune_0.2"
#peft_model="/vepfs/hongcheng/moe/MoE_unsupervised_pruning/result/train_5000_valid_500/baseline_finetuning"
#prune_result="."
batch_size=8
CUDA_VISIBLE_DEVICES="2,3" python main.py \
     --save_dir $save_dir \
     --data_dir $data_dir \
     --model_path $model_path \
     --peft_model $peft_model \
     --prune_result $prune_result \
		 --batch_size $batch_size

