CUDA_VISIBLE_DEVICES=1 nohup python3 tta.py --config_file tta_configs_rerun2/ham_rstp_tta/exp21/exp0.0.0.yaml > logs_rerun2/ham_rstp_tta/exp21/exp0.0.0.log 2>&1 &
wait
CUDA_VISIBLE_DEVICES=1 nohup python3 tta.py --config_file tta_configs_rerun2/ham_rstp_tta/exp21/exp0.0.1.yaml > logs_rerun2/ham_rstp_tta/exp21/exp0.0.1.log 2>&1 &
wait
CUDA_VISIBLE_DEVICES=1 nohup python3 tta.py --config_file tta_configs_rerun2/ham_rstp_tta/exp21/exp0.0.2.yaml > logs_rerun2/ham_rstp_tta/exp21/exp0.0.2.log 2>&1 &
wait