"""
高质量合成情绪数据生成器

基于生理学研究设计的严格分离规则：
- stressed: 高心率 + 低HRV（交感神经激活）
- sad: 低心率 + 低HRV（副交感神经抑制）
- neutral: 正常心率 + 中等HRV
- happy: 中等心率 + 高HRV（副交感神经活跃）

类别间特征范围严格分离，避免重叠导致的混淆。
"""

import numpy as np
import json
from pathlib import Path
from typing import List, Dict
from collections import Counter


# 生理特征范围定义（基于文献）
# 关键设计：RMSSD 作为主区分特征，各类别完全不重叠
EMOTION_RULES = {
    "stressed": {
        "hr":    {"min": 88,  "max": 110, "noise": 2},    # 交感激活，心率升高
        "rmssd": {"min": 10,  "max": 22,  "noise": 1.5},  # HRV 极低（上限22，与sad下限24分离）
        "sdnn":  {"min": 15,  "max": 30,  "noise": 1.5},  # SDNN 低
        "pnn50": {"min": 3,   "max": 12,  "noise": 1},    # PNN50 极低
    },
    "sad": {
        "hr":    {"min": 50,  "max": 64,  "noise": 2},    # 心率偏低
        "rmssd": {"min": 24,  "max": 33,  "noise": 1.5},  # HRV 低（下限24，与stressed上限22分离）
        "sdnn":  {"min": 25,  "max": 40,  "noise": 2},    # SDNN 低
        "pnn50": {"min": 5,   "max": 16,  "noise": 1},    # PNN50 偏低
    },
    "neutral": {
        "hr":    {"min": 62,  "max": 78,  "noise": 2},    # 正常心率
        "rmssd": {"min": 35,  "max": 48,  "noise": 2},    # HRV 中等（上限48，与happy下限50分离）
        "sdnn":  {"min": 42,  "max": 58,  "noise": 2},    # SDNN 中等
        "pnn50": {"min": 18,  "max": 30,  "noise": 1.5},  # PNN50 中等
    },
    "happy": {
        "hr":    {"min": 70,  "max": 85,  "noise": 2},    # 心率稍高但不紧张
        "rmssd": {"min": 50,  "max": 75,  "noise": 2.5},  # HRV 高（下限50，与neutral上限48分离）
        "sdnn":  {"min": 55,  "max": 85,  "noise": 2.5},  # SDNN 高
        "pnn50": {"min": 30,  "max": 50,  "noise": 2},    # PNN50 高
    },
}

# 4个类别的简单标签
EMOTION_LABELS = {
    "stressed": "stressed",
    "sad": "sad",
    "neutral": "neutral",
    "happy": "happy",
}

INSTRUCTION = "Based on the following physiological data, classify the user's emotional state as one of: happy, stressed, sad, neutral."


def generate_sample(emotion: str, rng: np.random.Generator) -> Dict:
    """为指定情绪生成一个样本"""
    rules = EMOTION_RULES[emotion]

    def sample_feature(rule):
        center = rng.uniform(rule["min"], rule["max"])
        noise = rng.normal(0, rule["noise"])
        value = center + noise
        # 严格裁剪到定义范围，避免类别间重叠
        return max(rule["min"], min(rule["max"], value))

    hr = sample_feature(rules["hr"])
    rmssd = sample_feature(rules["rmssd"])
    sdnn = sample_feature(rules["sdnn"])
    pnn50 = sample_feature(rules["pnn50"])

    # SDNN 应 >= RMSSD（生理约束）
    if sdnn < rmssd:
        sdnn = rmssd + rng.uniform(5, 15)
        sdnn = min(sdnn, rules["sdnn"]["max"])

    # PNN50 裁剪
    pnn50 = max(rules["pnn50"]["min"], min(rules["pnn50"]["max"], pnn50))

    input_text = (
        f"Heart Rate: {hr:.1f} bpm. "
        f"RMSSD: {rmssd:.1f} ms. "
        f"SDNN: {sdnn:.1f} ms. "
        f"PNN50: {pnn50:.1f}%."
    )

    return {
        "instruction": INSTRUCTION,
        "input": input_text,
        "output": EMOTION_LABELS[emotion],
        # 保存原始数值用于验证
        "_meta": {
            "hr": round(hr, 1),
            "rmssd": round(rmssd, 1),
            "sdnn": round(sdnn, 1),
            "pnn50": round(pnn50, 1),
            "emotion": emotion,
        }
    }


def generate_dataset(
    samples_per_class: int = 1250,
    seed: int = 42
) -> List[Dict]:
    """生成完整数据集"""
    rng = np.random.default_rng(seed)
    data = []

    for emotion in EMOTION_RULES:
        for _ in range(samples_per_class):
            sample = generate_sample(emotion, rng)
            data.append(sample)

    # 打乱顺序
    rng.shuffle(data)
    return data


def validate_data(data: List[Dict]) -> bool:
    """验证数据质量"""
    emotions = [d["_meta"]["emotion"] for d in data]
    counts = Counter(emotions)
    print("类别分布:")
    for e, c in sorted(counts.items()):
        print(f"  {e}: {c}")

    # 检查各类别特征范围
    print("\n各类别特征范围:")
    for emotion in EMOTION_RULES:
        subset = [d["_meta"] for d in data if d["_meta"]["emotion"] == emotion]
        hrs = [d["hr"] for d in subset]
        rmssds = [d["rmssd"] for d in subset]
        print(f"  {emotion}:")
        print(f"    HR:    [{min(hrs):.1f}, {max(hrs):.1f}]")
        print(f"    RMSSD: [{min(rmssds):.1f}, {max(rmssds):.1f}]")

    # 检查 RMSSD 是否有重叠
    ranges = {}
    for emotion in EMOTION_RULES:
        subset = [d["_meta"]["rmssd"] for d in data if d["_meta"]["emotion"] == emotion]
        ranges[emotion] = (min(subset), max(subset))

    print("\nRMSSD 范围重叠检查:")
    emotions_list = list(ranges.keys())
    has_overlap = False
    for i in range(len(emotions_list)):
        for j in range(i + 1, len(emotions_list)):
            e1, e2 = emotions_list[i], emotions_list[j]
            r1, r2 = ranges[e1], ranges[e2]
            overlap = r1[0] < r2[1] and r2[0] < r1[1]
            if overlap:
                print(f"  WARNING: {e1} 和 {e2} RMSSD 有重叠!")
                has_overlap = True
    if not has_overlap:
        print("  无重叠，数据质量合格")

    return not has_overlap


def save_data(data: List[Dict], output_dir: str):
    """保存数据（去除 _meta 字段）"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存完整数据（含 meta 用于调试）
    with open(output_dir / "full_data_with_meta.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # 保存不含 meta 的训练数据
    train_data = [{k: v for k, v in d.items() if k != "_meta"} for d in data]
    with open(output_dir / "full_data.json", "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)

    print(f"已保存 {len(data)} 条数据到 {output_dir}")


def split_dataset(data: List[Dict], output_dir: str, seed: int = 42):
    """分层划分数据集"""
    from sklearn.model_selection import train_test_split

    output_dir = Path(output_dir)

    # 去除 meta
    clean_data = [{k: v for k, v in d.items() if k != "_meta"} for d in data]
    labels = [d["_meta"]["emotion"] for d in data]

    # 分层划分
    train_val, test = train_test_split(
        clean_data, test_size=0.15, random_state=seed, stratify=labels
    )
    train_val_labels = [d["output"] for d in train_val]
    train, val = train_test_split(
        train_val, test_size=0.176, random_state=seed, stratify=train_val_labels  # 0.176 * 0.85 ≈ 0.15
    )

    for name, dataset in [("train_split", train), ("val_split", val), ("test_split", test)]:
        path = output_dir / f"{name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"训练集: {len(train)} | 验证集: {len(val)} | 测试集: {len(test)}")

    # 检查各集合的类别分布
    for name, dataset in [("train", train), ("val", val), ("test", test)]:
        dist = Counter(d["output"] for d in dataset)
        print(f"  {name}: {dict(sorted(dist.items()))}")

    return train, val, test


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="生成高质量情绪识别训练数据")
    parser.add_argument("--samples_per_class", type=int, default=1250, help="每类样本数")
    parser.add_argument("--output_dir", type=str, default="./data/processed", help="输出目录")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    print("=" * 60)
    print("生成高质量情绪识别训练数据")
    print("=" * 60)

    # 生成数据
    data = generate_dataset(args.samples_per_class, args.seed)
    print(f"\n生成 {len(data)} 条数据")

    # 验证数据质量
    print("\n" + "-" * 40)
    validate_data(data)

    # 保存和划分
    print("\n" + "-" * 40)
    save_data(data, args.output_dir)
    split_dataset(data, args.output_dir, args.seed)

    print("\n数据生成完成！")
