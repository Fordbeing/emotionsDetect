"""
一键执行完整流水线

数据生成 -> 训练 -> 评估 -> 可视化报告
"""

import subprocess
import sys
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent
SCRIPTS_DIR = Path(__file__).parent


def run_step(name, cmd, cwd=None):
    """执行一个步骤"""
    print("\n" + "=" * 60)
    print(f"  {name}")
    print("=" * 60)
    print(f"  命令: {' '.join(cmd)}")
    print("-" * 60)

    result = subprocess.run(cmd, cwd=cwd or str(PROJECT_ROOT))

    if result.returncode != 0:
        print(f"\n  {name} 失败！(exit code: {result.returncode})")
        sys.exit(1)

    print(f"\n  {name} 完成！")
    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(description="一键执行完整流水线")
    parser.add_argument("--skip_data", action="store_true", help="跳过数据生成")
    parser.add_argument("--skip_train", action="store_true", help="跳过训练")
    parser.add_argument("--skip_eval", action="store_true", help="跳过评估")
    parser.add_argument("--samples_per_class", type=int, default=1250)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--no_4bit", action="store_true")
    args = parser.parse_args()

    python = sys.executable

    print("=" * 60)
    print("  情绪识别完整流水线")
    print("  Qwen2.5-7B-Instruct + int4 + LoRA")
    print("=" * 60)

    # Step 1: 数据生成
    if not args.skip_data:
        run_step("Step 1: 数据生成", [
            python, str(SCRIPTS_DIR / "generate_data.py"),
            "--samples_per_class", str(args.samples_per_class),
            "--output_dir", str(PROJECT_ROOT / "data" / "processed"),
        ])
    else:
        print("\n跳过数据生成")

    # Step 2: 训练
    if not args.skip_train:
        train_cmd = [
            python, str(SCRIPTS_DIR / "train_qwen.py"),
            "--train_data", str(PROJECT_ROOT / "data" / "processed" / "train_split.json"),
            "--val_data", str(PROJECT_ROOT / "data" / "processed" / "val_split.json"),
            "--output_dir", str(PROJECT_ROOT / "models" / "qwen_lora"),
            "--epochs", str(args.epochs),
        ]
        if args.no_4bit:
            train_cmd.append("--no_4bit")
        run_step("Step 2: LoRA 训练", train_cmd)
    else:
        print("\n跳过训练")

    # Step 3: 评估
    if not args.skip_eval:
        eval_cmd = [
            python, str(SCRIPTS_DIR / "evaluate_qwen.py"),
            "--test_data", str(PROJECT_ROOT / "data" / "processed" / "test_split.json"),
            "--adapter_path", str(PROJECT_ROOT / "models" / "qwen_lora"),
            "--output_file", str(PROJECT_ROOT / "evaluation_results_v2.json"),
        ]
        if args.no_4bit:
            eval_cmd.append("--no_4bit")
        run_step("Step 3: 模型评估", eval_cmd)
    else:
        print("\n跳过评估")

    # Step 4: 可视化报告
    run_step("Step 4: 生成报告", [
        python, str(SCRIPTS_DIR / "plot_results.py"),
        "--results", str(PROJECT_ROOT / "evaluation_results_v2.json"),
        "--model_dir", str(PROJECT_ROOT / "models" / "qwen_lora"),
        "--output_dir", str(PROJECT_ROOT / "reports"),
    ])

    # 完成
    print("\n" + "=" * 60)
    print("  流水线全部完成！")
    print("=" * 60)
    print(f"  LoRA adapter:  {PROJECT_ROOT / 'models' / 'qwen_lora'}")
    print(f"  评估结果:      {PROJECT_ROOT / 'evaluation_results_v2.json'}")
    print(f"  可视化报告:    {PROJECT_ROOT / 'reports'}")
    print()


if __name__ == "__main__":
    main()
