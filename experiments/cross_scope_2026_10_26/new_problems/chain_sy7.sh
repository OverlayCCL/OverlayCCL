#!/bin/bash
# 7-node (224-rank) campaign: A) e2e Llama (amp5, GQA=4, L=2, M=4) baseline vs OverlayCCL arms; B) per-problem training for the
# 10 new problems (baseline + best hand-written + gate winner); C) 224-rank forward bench; D) 5-node / 3-node e2e subsets; E) 2nd seed.
G=/home/ubuntu/g2; cd /home/ubuntu/OverlayCCL; source /home/ubuntu/venv.env
export RESULTS_DIR=$G/sy7_e2e PERCALL_PROBE=0; mkdir -p $RESULTS_DIR $G/sy7_e2e/json
R() { tag=$1; shift; echo "=== [$(date -u +%T)] $tag"; timeout 2400 bash experiments/cluster_224rank/scripts/run_ap7.sh experiments/model_extension/train_llama_e2e_amp5.py "$@" > $G/sy7_e2e/$tag.out 2>&1; grep -h "\[bench\]" $G/sy7_e2e/logs/*${tag}*node*.log 2>/dev/null | head -1; }
C="N_LAYERS_PER_STAGE=2 N_MB=4 GQA=4"
R e2e_base_per_tensor per_mb 300 e2e_base_per_tensor GRAD_SYNC=per_tensor $C
R e2e_bund_per_tensor bundled 300 e2e_bund_per_tensor GRAD_SYNC=per_tensor $C
R e2e_bund_buckets_8 bundled 300 e2e_bund_buckets_8 GRAD_SYNC=buckets_8 $C
R e2e_bund_flat bundled 300 e2e_bund_flat GRAD_SYNC=flat $C
R e2e_bund_per_layer bundled 300 e2e_bund_per_layer GRAD_SYNC=per_layer $C
R e2e_mixed_base per_mb 300 e2e_mixed_base GRAD_SYNC=per_tensor MIXED=1 $C
R e2e_mixed_bund_per_tensor bundled 300 e2e_mixed_bund_per_tensor GRAD_SYNC=per_tensor MIXED=1 $C
R e2e_mixed_bund_bf16_flat bundled 300 e2e_mixed_bund_bf16_flat GRAD_SYNC=bf16_flat MIXED=1 $C
echo SY7_A_DONE
# B) per-problem 224-rank training (bench32 on the master + training on all 224 ranks)
cp $G/g_newprobs_sy7.py $G/g_newprobs.py; cp $G/g_train_v4.py $G/g_train.py
PROBS='hs_a_64x64k_512k probe_het_48_small layer_grads_hetero_ar probe_het_24_mid hs_c_32x256k_2m hetero_grad_ar_many model_grads_4layers probe_het_48_mid zero2_hetero_rs' RUN_TIMEOUT=1500 NP_OUT=$G/sy7_train bash $G/g_newprobs_run2.sh > $G/sy7_train.out 2>&1
cp $G/g_train_v5.py $G/g_train.py
PROBS='layer_grads_mixed_dtype' RUN_TIMEOUT=1500 NP_OUT=$G/sy7_train_dtype bash $G/g_newprobs_run2.sh > $G/sy7_train_dtype.out 2>&1
cp $G/g_train_v4.py $G/g_train.py
echo SY7_B_DONE
# C) 224-rank forward bench
PROBS='hs_a_64x64k_512k probe_het_48_small layer_grads_hetero_ar probe_het_24_mid hs_c_32x256k_2m hetero_grad_ar_many model_grads_4layers probe_het_48_mid zero2_hetero_rs layer_grads_mixed_dtype' B64_OUT=$G/sy7_bench224 bash $G/g_bench64.sh > $G/sy7_bench224.out 2>&1
echo SY7_C_DONE
# D) cluster-size subsets (5 nodes = 4 workers, 3 nodes = 2 workers); HID and VOCAB scale with world size
W5=$(echo $WORKERS | awk '{print $1,$2,$3,$4}'); W3=$(echo $WORKERS | awk '{print $1,$2}')
for cfg in "5n:$W5" "3n:$W3"; do n=${cfg%%:*}; w=${cfg#*:}
  WORKERS_OVERRIDE="$w" R e2e_${n}_base per_mb 300 e2e_${n}_base GRAD_SYNC=per_tensor $C
  WORKERS_OVERRIDE="$w" R e2e_${n}_bund bundled 300 e2e_${n}_bund GRAD_SYNC=per_tensor $C
done
echo SY7_D_DONE
# E) second seed of the two headline arms
R e2e_base_per_tensor_s1 per_mb 300 e2e_base_per_tensor_s1 GRAD_SYNC=per_tensor $C --seed 1
R e2e_bund_per_tensor_s1 bundled 300 e2e_bund_per_tensor_s1 GRAD_SYNC=per_tensor $C --seed 1
cp /tmp/tp_search/llama_e2e_amp5_*.json $G/sy7_e2e/json/ 2>/dev/null
for w in $WORKERS; do scp -q -i $SSH_KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR "ubuntu@$w:/tmp/tp_search/llama_e2e_amp5_*.json" $G/sy7_e2e/json/ 2>/dev/null; done
echo SY7_ALL_DONE
