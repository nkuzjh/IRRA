#!/bin/bash

# 设置基础参数
# num_gpus=1
task_name="mllm_rstp_tta_best2"
log_dir="./logs_rerun"
mkdir -p $log_dir/$task_name



# 定义实验名
exp_names=(
    "exp3.0.0"
    "exp3.0.1"
    "exp3.0.2"
    "exp3.0.3"
    "exp3.0.4"
    "exp3.1.0"
    "exp3.1.1"
    "exp3.1.2"
    "exp3.1.3"
    "exp3.1.4"
    "exp3.2.0"
    "exp3.2.1"
    "exp3.2.2"
    "exp3.2.3"
    "exp3.2.4"
)

# 按顺序执行每个训练任务
for i in "${!exp_names[@]}"
do
    exp_name=${exp_names[$i]}

    echo " "
    echo "Starting Training: $task_name $exp_name ..."
    start_time=$(date +%s)
    echo "Start Time: $(date +"%Y-%m-%d %T")"


    # CUDA_VISIBLE_DEVICES=1 nohup python3 tta.py --config_file tta_configs/mllm_rstp_tta/exp1.yaml > logs/mllm_rstp_tta/exp1.log 2>&1 &
    CUDA_VISIBLE_DEVICES=0 nohup python3 tta.py --config_file tta_configs/$task_name/$exp_name.yaml > $log_dir/$task_name/$exp_name.log 2>&1 &


    # 等待当前任务完成
    wait

    # 记录结束时间
    end_time=$(date +%s)
    duration=$(( end_time - start_time ))
    # 格式化时间（分钟和秒）
    minutes=$(( duration / 60 ))
    seconds=$(( duration % 60 ))
    # 结束时间和运行时长
    echo "End Time: $(date +"%Y-%m-%d %T")"
    echo "Duration: ${minutes}m${seconds}s"
    echo "Finsh Training $task_name $exp_name ."
done
