#!/bin/bash

# 设置基础参数
# num_gpus=1
task_name="ham_icfg_tta_best2"
log_dir="./logs"
mkdir -p $log_dir/$task_name



# 定义实验名
exp_names=(
    "exp2.0.0"
    "exp2.0.1"
    "exp2.0.2"
    "exp2.0.3"
    "exp2.0.4"
    "exp2.1.0"
    "exp2.1.1"
    "exp2.1.2"
    "exp2.1.3"
    "exp2.1.4"
)

# 按顺序执行每个训练任务
for i in "${!exp_names[@]}"
do
    exp_name=${exp_names[$i]}

    echo " "
    echo "Starting Training: $task_name $exp_name ..."
    start_time=$(date +%s)
    echo "Start Time: $(date +"%Y-%m-%d %T")"


    # nohup python3 tta.py --config_file tta_configs/ham_icfg_tta/exp1.yaml > logs/ham_icfg_tta/exp1.log 2>&1 &
    CUDA_VISIBLE_DEVICES=2 nohup python3 tta.py --config_file tta_configs/$task_name/$exp_name.yaml > logs/$task_name/$exp_name.log 2>&1 &


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
