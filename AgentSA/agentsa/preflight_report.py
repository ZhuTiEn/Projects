"""Render the completed preflight outputs into a Chinese report and ranking plot."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .utils import read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs/preflight_20261102")
    args = parser.parse_args()
    root = Path(args.output).resolve()
    completion = read_json(root / "completion_all.json")
    search = read_json(root / "clean_search/selection.json")
    ranks = read_json(root / "pilot_rankings/summary.json")
    lines = ["# AgentSA：正式实验前检查报告", "",
        "范围：只训练真实数据上的干净推荐模型，并重新评估已完成的 pilot 模型。没有生成新攻击数据。", "",
        "## 1. 验证集选参", "",
        "预先固定学习率网格：NMF 0.001/0.0003；NeuNMF 0.001/0.0003/0.0001。",
        "每组使用种子 11、22、33，最多 100 轮，早停耐心值 10，其余模型结构、正则与批量设置不变。",
        "按三个种子的最佳验证 RMSE 均值选择；测试集和攻击指标均不参与选参。", "",
        "| 模型 | 学习率 | 验证 RMSE 均值 | 样本标准差 | 最佳轮次（11/22/33） | 达到100轮上限的次数 |",
        "|---|---:|---:|---:|---|---:|"]
    for row in search["candidates"]:
        lines.append(f"| {row['model']} | {row['learning_rate']:g} | {row['validation_rmse_mean']:.6f} | "
                     f"{row['validation_rmse_std']:.6f} | {row['best_epochs']} | {row['runs_at_epoch_cap']} |")
    lines += ["", "在此次有限网格内选择："]
    for name, selection in search["selected"].items():
        original = next(g for g in search["candidates"] if g["model"] == name and g["learning_rate"] == 0.001)
        gain = original["validation_rmse_mean"] - selection["validation_rmse_mean"]
        lines.append(f"- {name}：学习率 {selection['learning_rate']:g}，验证 RMSE 均值 "
                     f"{selection['validation_rmse_mean']:.6f}，相对原学习率降低 {gain:.6f}。")
    lines += ["", "这是局部学习率检查，不是完整超参数优化，也没有进行显著性检验。",
        "若出现达到 100 轮上限且仍持续改善的情况，应扩大训练上限后重新判断，不能宣称充分收敛。", "",
        "配置保存在 selected_configs/nmf.yaml 和 selected_configs/neunmf.yaml，二者可分别用于后续模型运行。",
        "这些配置保留 pilot 的 3 个假用户等攻击设置，只冻结了本次选出的训练设置，不是 5% 正式攻击配置。", "",
        "![验证集曲线](clean_search/validation_curves.png)", "", "## 2. 原 pilot 排名诊断", "",
        "以下全部使用原 pilot 的旧模型参数和 checkpoint，不是使用新选参数重跑攻击。",
        "评估目标 1920；只包括训练中未评价该目标的真实用户；全商品候选、屏蔽训练已见商品、同分按商品 ID 升序。",
        "HR@10 仍是主要曝光指标；HR@50/100 及排名用于解释零曝光。", "",
        "| 模型 | 方法 | 用户数 | 排名P10 | 排名中位数 | 排名P90 | HR@10 | HR@50 | HR@100 | Top10分差中位数 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in ranks:
        lines.append(f"| {r['model']} | {r['method']} | {r['users']} | {r['rank_p10']:.1f} | "
            f"{r['rank_p50']:.1f} | {r['rank_p90']:.1f} | {r['hr_at_10']:.3%} | {r['hr_at_50']:.3%} | "
            f"{r['hr_at_100']:.3%} | {r['score_gap_top_10_median']:.6f} |")
    lines += ["", "Top10 分差 = 目标预测评分 − 该用户第10名候选评分；负数表示低于门槛。",
        "分差为零可能是同分，是否命中仍按含 ID 平局规则的实际排名判断。", "",
        "| 模型 | 方法 | 平均排名改善 | 排名改善中位数 | 改善用户比例 | 不变比例 | 恶化比例 |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for r in ranks:
        if r["method"] != "clean":
            lines.append(f"| {r['model']} | {r['method']} | {r['rank_improvement_mean']:.2f} | "
                f"{r['rank_improvement_median']:.1f} | {r['users_improved_fraction']:.2%} | "
                f"{r['users_unchanged_fraction']:.2%} | {r['users_worsened_fraction']:.2%} |")
    lines += ["", "排名改善 = 干净排名 − 投毒排名，正值表示上升。平均数、中位数及改善比例表达不同方面，需一起看。", ""]
    for model in ("nmf", "neunmf"):
        clean = next(r for r in ranks if r["model"] == model and r["method"] == "clean")
        agent = next(r for r in ranks if r["model"] == model and r["method"] == "agent")
        lines.append(f"- {model}：干净模型的目标排名中位数 {clean['rank_p50']:.0f}，Agent 投毒后 "
            f"{agent['rank_p50']:.0f}；逐用户排名改善均值 {agent['rank_improvement_mean']:.2f}，"
            f"改善用户比例 {agent['users_improved_fraction']:.2%}。Top10 分差中位数从 "
            f"{clean['score_gap_top_10_median']:.4f} 变为 {agent['score_gap_top_10_median']:.4f}。")
    if all(r["hr_at_100"] == 0 for r in ranks):
        lines += ["", "本次所有方法连 HR@100 也为零：目标整体距前排较远。评分上升尚未转化为前100名曝光；不能把 HR@10 为零归因于显示精度或只差一点点跨过前10名。"]
    lines += ["",
        "各方法的 HR@10 和有符号 Prediction Shift 已与原 results.csv 重新核对一致。", "",
        "![目标排名累积分布](pilot_rankings/rank_cdf.png)", "", "## 3. 产物与复现", "",
        "- clean_search/runs.csv：15 次干净训练的逐次记录。",
        "- clean_search/protocol.json：选参范围、规则、配置和代码哈希。",
        "- clean_search/selection.json：各组统计与参数选择。",
        "- pilot_rankings/summary.csv：10 个原 pilot 模型的排名汇总。",
        "- pilot_rankings/<模型>/<种子>/target_1920/<方法>/per_user.csv：逐用户排名、评分和门槛分差。",
        "- 对应 per_user.npz：同一份逐用户结果的压缩数组。",
        "- completion_all.json：完成状态和原 pilot/Agent 文件的哈希核对。", "",
        f"原 pilot 和 Agent 文件内容保持不变：{completion['original_files_unchanged']}。本次新增 API 调用：0。",
        f"训练搜索与排名诊断耗时：{completion['elapsed_seconds']/60:.2f} 分钟。", "",
        "```powershell",
        "Set-Location C:\\Projects\\AgentSA",
        ".venv\\Scripts\\python.exe -m agentsa.preflight --stage all",
        ".venv\\Scripts\\python.exe -m agentsa.preflight_report",
        "```", "",
        "重复运行会复用配置匹配的干净模型和诊断缓存；改变搜索规则、源代码或配置时应使用新 --output 目录。",
        "前置检查输出目录和旧 pilot 目录不能由多个进程同时写入。", "",
        "## 4. 后续范围", "",
        "先依据本报告确定模型参数。下一阶段才是单目标、992 个假用户的正式规模试跑。",
        "正式规模需要独立配置、Agent 输出目录、明确的评分预算协议和请求/token 预算；本次没有启动该阶段。",
        "这次仍只有一个目标、一个攻击生成种子，原 pilot 排名结果不能证明方法普遍优劣。", ""]
    plot_rankings(root, ranks)
    (root / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(root / "REPORT.md")


def plot_rankings(root, ranks):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for ax, model in zip(axes, ("nmf", "neunmf")):
        for r in ranks:
            if r["model"] != model:
                continue
            path = root / "pilot_rankings" / model / str(r["seed"]) / f"target_{r['target']}" / r["method"] / "per_user.npz"
            with np.load(path, allow_pickle=False) as data:
                values = np.sort(data["ranks"])
            ax.step(values, np.arange(1, len(values)+1) / len(values), where="post", label=r["method"])
        ax.set(title=f"{model.upper()}: target 1920 (original pilot)",
               xlabel="Target rank (smaller is better)", ylabel="Fraction of eligible users")
        ax.set_xlim(left=0)
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(alpha=0.2)
    fig.savefig(root / "pilot_rankings/rank_cdf.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
