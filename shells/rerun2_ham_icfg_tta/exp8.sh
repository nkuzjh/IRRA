#!/bin/bash

# 设置基础参数
# num_gpus=1
task_name="ham_icfg_tta/exp8"
log_dir="./logs_rerun2"
mkdir -p $log_dir/$task_name



# 定义实验名
exp_names=(
    "exp0"
    "exp0.0.1"
    "exp0.0.2"
    "exp0.0.3"
    "exp0.0.4"

    "exp0.1"
    "exp0.1.1"
    "exp0.1.2"
    "exp0.1.3"
    "exp0.1.4"

    "exp0.2"
    "exp0.2.1"
    "exp0.2.2"
    "exp0.2.3"
    "exp0.2.4"

    "exp0.3"
    "exp0.3.1"
    "exp0.3.2"
    "exp0.3.3"
    "exp0.3.4"

    "exp0.4"
    "exp0.4.1"
    "exp0.4.2"
    "exp0.4.3"
    "exp0.4.4"

    "exp0.5"
    "exp0.5.1"
    "exp0.5.2"
    "exp0.5.3"
    "exp0.5.4"
)

# 按顺序执行每个训练任务
for i in "${!exp_names[@]}"
do
    exp_name=${exp_names[$i]}

    echo " "
    echo "Starting Training: $task_name $exp_name ..."
    start_time=$(date +%s)
    echo "Start Time: $(date +"%Y-%m-%d %T")"


    # nohup python3 tta.py --config_file tta_configs/ham_rstp_tta/exp0.0.0.yaml > logs/ham_rstp_tta/exp0.0.0.log 2>&1 &
    #CUDA_VISIBLE_DEVICES=2
    CUDA_VISIBLE_DEVICES=0 nohup python3 tta.py --config_file tta_configs_rerun2/$task_name/$exp_name.yaml > $log_dir/$task_name/$exp_name.log 2>&1 &


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
