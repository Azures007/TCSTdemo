#!/usr/bin/env bash
set -uo pipefail

cd "$HOME/autodl-tmp/TCSTdemo/hfcst_work" || exit 1
export DATA_ROOT="$HOME/autodl-tmp/v7.0/PTMDA-Net_v7_reopt/Datasets"
: "${STAMP:?缺少实验时间标识}"

failed=0

for name in houston hyrank shanghai_hangzhou; do
    cfg="configs_hfcst/${name}_replay_best_${STAMP}.yaml"
    log="logs_hfcst/${name}_replay_best_${STAMP}.log"

    echo
    echo "========== 开始：${name} =========="
    echo "配置：${cfg}"
    echo "日志：${log}"

    if python -u scripts/train_hfcst.py \
        --config "$cfg" \
        --device cuda:0 2>&1 | tee "$log"; then
        echo "[完成] ${name}"
    else
        echo "[失败] ${name}，继续运行剩余数据集"
        failed=1
    fi
done

echo
echo "========== 三个数据集运行结束 =========="
echo "请检查各自日志中的最终 OA、AA、Macro-F1。"
exit "$failed"
