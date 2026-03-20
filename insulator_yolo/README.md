# Insulator YOLO

面向 IDID 绝缘子缺陷数据集的可复用 Ultralytics YOLO 工程。

## 工作流

1. 准备数据集
   - `PYTHONPATH=src python3 scripts/prepare_dataset.py --config configs/dataset.yaml`
2. 训练
   - `PYTHONPATH=src python3 scripts/train.py --config configs/train.yaml`
   - 可以在 `configs/train.yaml` 中直接补充 Ultralytics 训练参数，例如 `amp`、`patience`、`cache`、`cos_lr`。
3. 验证
   - `PYTHONPATH=src python3 scripts/validate.py --config configs/train.yaml --weights <weights>`
4. 推理
   - `PYTHONPATH=src python3 scripts/predict.py --config configs/predict.yaml`
   - `configs/predict.yaml` 支持直接透传 Ultralytics 推理参数，例如 `classes`、`iou`、`max_det`。

## 对比可视化

当你希望对带标注样本做人工核查时，可以使用对比可视化工具生成三栏图：

1. 原图
2. 真实标注
3. 模型预测

默认行为：

- 读取已经准备好的 YOLO 数据集
- 默认使用 `val` 划分
- 按固定随机种子稳定抽样 `20` 张图片
- 默认只显示缺陷类，不显示 `normal_insulator`
- 输出写入 `artifacts/runs/comparisons/`

示例：

```bash
PYTHONPATH=src python3 scripts/visualize_comparison.py --config configs/compare.yaml
```

常见覆盖参数：

```bash
PYTHONPATH=src python3 scripts/visualize_comparison.py \
  --config configs/compare.yaml \
  --split train \
  --limit 10 \
  --weights /path/to/best.pt \
  --save-dir artifacts/runs/comparisons/manual_check
```

这个流程适合做人工定性检查，但不能替代 `mAP` 这类定量评估指标。

## 产物目录

- 处理后的数据集：`artifacts/processed/`
- 训练与推理输出：`artifacts/runs/`
