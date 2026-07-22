"""
评估结果可视化脚本

生成混淆矩阵热力图、各类别指标柱状图、训练 loss 曲线。
"""

import json
import sys
import os
from pathlib import Path

import numpy as np


def plot_confusion_matrix(cm, labels, output_path):
    """绘制混淆矩阵热力图"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("  跳过混淆矩阵（需要 matplotlib 和 seaborn）")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    cm_array = np.array(cm)

    sns.heatmap(
        cm_array, annot=True, fmt='d', cmap='Blues',
        xticklabels=labels, yticklabels=labels, ax=ax
    )
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  混淆矩阵已保存: {output_path}")


def plot_metrics_bar(report, labels, output_path):
    """绘制各类别指标柱状图"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  跳过指标图（需要 matplotlib）")
        return

    metrics = ['precision', 'recall', 'f1-score']
    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, metric in enumerate(metrics):
        values = [report.get(label, {}).get(metric, 0) for label in labels]
        bars = ax.bar(x + i * width, values, width, label=metric.capitalize())
        # 在柱子上标注数值
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Emotion', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Classification Metrics by Emotion', fontsize=14)
    ax.set_xticks(x + width)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  指标图已保存: {output_path}")


def plot_training_loss(log_dir, output_path):
    """从 Trainer 日志绘制训练 loss 曲线"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  跳过 loss 曲线（需要 matplotlib）")
        return

    # 尝试从 trainer_state.json 读取
    state_file = Path(log_dir) / "trainer_state.json"
    if not state_file.exists():
        # 搜索子目录
        for p in Path(log_dir).rglob("trainer_state.json"):
            state_file = p
            break
        else:
            print(f"  未找到 trainer_state.json，跳过 loss 曲线")
            return

    with open(state_file) as f:
        state = json.load(f)

    log_history = state.get("log_history", [])

    train_steps = []
    train_losses = []
    eval_steps = []
    eval_losses = []

    for entry in log_history:
        if "loss" in entry and "eval_loss" not in entry:
            train_steps.append(entry["step"])
            train_losses.append(entry["loss"])
        if "eval_loss" in entry:
            eval_steps.append(entry["step"])
            eval_losses.append(entry["eval_loss"])

    fig, ax = plt.subplots(figsize=(10, 6))

    if train_steps:
        ax.plot(train_steps, train_losses, 'b-', label='Train Loss', alpha=0.7)
    if eval_steps:
        ax.plot(eval_steps, eval_losses, 'ro-', label='Eval Loss', markersize=8)

    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training & Evaluation Loss', fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Loss 曲线已保存: {output_path}")


def generate_text_report(results, output_path):
    """生成文本评估报告"""
    labels = ["happy", "stressed", "sad", "neutral"]

    lines = []
    lines.append("=" * 60)
    lines.append("情绪识别模型评估报告")
    lines.append("=" * 60)
    lines.append(f"准确率 (Accuracy): {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    lines.append(f"Macro F1:          {results['macro_f1']:.4f}")
    lines.append(f"Weighted F1:       {results['weighted_f1']:.4f}")
    lines.append("")

    report = results.get("classification_report", {})
    lines.append("各类别详细指标:")
    lines.append("-" * 50)
    for label in labels:
        if label in report:
            m = report[label]
            lines.append(f"  {label}:")
            lines.append(f"    Precision: {m['precision']:.4f}")
            lines.append(f"    Recall:    {m['recall']:.4f}")
            lines.append(f"    F1-Score:  {m['f1-score']:.4f}")
            lines.append(f"    Support:   {m['support']}")

    # 混淆矩阵
    if "confusion_matrix" in results:
        lines.append("")
        lines.append("混淆矩阵:")
        lines.append("         pred: " + "  ".join(f"{l:>8}" for l in labels))
        cm = results["confusion_matrix"]
        for i, row in enumerate(cm):
            lines.append(f"  true {labels[i]:>8}: " + "  ".join(f"{v:>8}" for v in row))

    # 错误分析
    if "predictions" in results and "true_labels" in results:
        errors = [(t, p) for t, p in zip(results["true_labels"], results["predictions"]) if t != p]
        lines.append("")
        lines.append(f"错误样本数: {len(errors)} / {len(results['true_labels'])}")

        from collections import Counter
        error_counts = Counter(errors)
        lines.append("最常见错误:")
        for (true, pred), count in error_counts.most_common(5):
            lines.append(f"  {true} -> {pred}: {count}次")

    # unknown 统计
    if "raw_outputs" in results:
        unknown_count = sum(1 for r in results.get("raw_outputs", []) if r == "unknown")
        if unknown_count > 0:
            lines.append(f"\n无法识别标签数: {unknown_count}")

    text = "\n".join(lines)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"  文本报告已保存: {output_path}")
    return text


def main():
    import argparse

    parser = argparse.ArgumentParser(description="评估结果可视化")
    parser.add_argument("--results", type=str, default="./evaluation_results_v2.json")
    parser.add_argument("--model_dir", type=str, default="./models/qwen_lora")
    parser.add_argument("--output_dir", type=str, default="./reports")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("生成评估报告")
    print("=" * 60)

    # 加载评估结果
    if not os.path.exists(args.results):
        print(f"错误: 评估结果文件不存在: {args.results}")
        print("请先运行: python scripts/evaluate_qwen.py")
        sys.exit(1)

    with open(args.results) as f:
        results = json.load(f)

    labels = ["happy", "stressed", "sad", "neutral"]

    # 1. 文本报告
    print("\n[1/4] 生成文本报告...")
    report_text = generate_text_report(results, output_dir / "evaluation_report.txt")
    print(report_text)

    # 2. 混淆矩阵
    print("\n[2/4] 绘制混淆矩阵...")
    if "confusion_matrix" in results:
        plot_confusion_matrix(results["confusion_matrix"], labels, output_dir / "confusion_matrix.png")

    # 3. 指标柱状图
    print("\n[3/4] 绘制指标柱状图...")
    if "classification_report" in results:
        plot_metrics_bar(results["classification_report"], labels, output_dir / "metrics_bar.png")

    # 4. 训练 loss 曲线
    print("\n[4/4] 绘制训练 loss 曲线...")
    plot_training_loss(args.model_dir, output_dir / "training_loss.png")

    print("\n" + "=" * 60)
    print(f"报告已生成到: {output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
