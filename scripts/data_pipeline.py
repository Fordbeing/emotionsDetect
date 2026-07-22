"""
数据管道 - 调用 generate_data 生成数据并划分数据集
"""

import sys
import json
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent))
from generate_data import generate_dataset, validate_data, save_data, split_dataset


def run_pipeline(
    samples_per_class: int = 1250,
    output_dir: str = "./data/processed",
    seed: int = 42,
):
    """运行完整数据管道"""
    print("=" * 60)
    print("情绪识别数据管道")
    print("=" * 60)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 生成数据
    print("\n[1/3] 生成合成数据...")
    data = generate_dataset(samples_per_class, seed)
    print(f"  生成 {len(data)} 条数据")

    # 2. 验证数据质量
    print("\n[2/3] 验证数据质量...")
    is_valid = validate_data(data)
    if not is_valid:
        print("  警告: 数据存在类别间重叠，但已继续")

    # 3. 保存和划分
    print("\n[3/3] 保存并划分数据集...")
    save_data(data, output_dir)
    split_dataset(data, output_dir, seed)

    print("\n" + "=" * 60)
    print("数据管道完成！")
    print(f"输出目录: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="情绪识别数据管道")
    parser.add_argument("--samples_per_class", type=int, default=1250)
    parser.add_argument("--output_dir", type=str, default="./data/processed")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_pipeline(args.samples_per_class, args.output_dir, args.seed)
