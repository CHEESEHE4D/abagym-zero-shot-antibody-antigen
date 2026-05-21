#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
22_plot_4models_results.py

四模型结果作图脚本（高辨识度配色版）
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Iterable, List
import pandas as pd
import matplotlib.pyplot as plt

MODEL_ORDER = ["ESM1v", "ESM2", "SaProt", "ESM-IF-AF2"]
VIRUS_ORDER = ["COVID-19", "HIV", "Influenza"]

# 高辨识度配色：浅蓝 / 橙红 / 青蓝 / 深 navy
COLORS = {
    "esm1v": "#7DB7D9",
    "esm2": "#D95F4A",
    "saprot": "#2A9D8F",
    "esmif": "#163A5F",
    "available": "#A9D3EA",
    "common": "#1F5A85",
    "grid": "#D9D9D9",
    "text": "#222222",
}

MODEL_COLORS = {
    "ESM1v": COLORS["esm1v"],
    "ESM2": COLORS["esm2"],
    "SaProt": COLORS["saprot"],
    "ESM-IF-AF2": COLORS["esmif"],
}

SET_COLORS = {
    "Available set": COLORS["available"],
    "Common set": COLORS["common"],
}

def setup_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei", "SimHei", "Arial Unicode MS",
        "Noto Sans CJK SC", "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["axes.edgecolor"] = "#333333"
    plt.rcParams["axes.labelcolor"] = COLORS["text"]
    plt.rcParams["xtick.color"] = COLORS["text"]
    plt.rcParams["ytick.color"] = COLORS["text"]
    plt.rcParams["text.color"] = COLORS["text"]

def beautify_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="-", linewidth=0.5, alpha=0.45, color=COLORS["grid"])
    ax.set_axisbelow(True)

def save_fig(fig: plt.Figure, out_base: Path, formats: Iterable[str]) -> None:
    for fmt in formats:
        fig.savefig(out_base.with_suffix(f".{fmt}"), bbox_inches="tight")
    plt.close(fig)

def reorder_models(df: pd.DataFrame, model_col: str = "model") -> pd.DataFrame:
    df = df.copy()
    if model_col in df.columns:
        order_map = {m: i for i, m in enumerate(MODEL_ORDER)}
        df["_model_order"] = df[model_col].map(order_map).fillna(999)
        df = df.sort_values(["_model_order", model_col]).drop(columns=["_model_order"])
    return df

def plot_model_coverage(eval_dir: Path, out_dir: Path, formats: List[str]) -> None:
    f = eval_dir / "model_score_coverage_summary.csv"
    if not f.exists():
        return
    df = reorder_models(pd.read_csv(f))
    y_col = "n_usable_rows" if "n_usable_rows" in df.columns else "n_rows"
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    bars = ax.bar(df["model"], df[y_col], color=[MODEL_COLORS.get(m, "#999999") for m in df["model"]],
                  edgecolor="white", linewidth=0.8)
    ax.set_xlabel("模型"); ax.set_ylabel("有效突变记录数"); ax.set_title("四模型有效预测覆盖情况")
    ax.tick_params(axis="x", rotation=20); beautify_axes(ax)
    for b, v in zip(bars, df[y_col]):
        ax.text(b.get_x()+b.get_width()/2, b.get_height(), str(int(v)), ha="center", va="bottom", fontsize=9)
    save_fig(fig, out_dir / "fig3_1_model_coverage", formats)

def plot_model_spearman_available_common(eval_dir: Path, out_dir: Path, formats: List[str]) -> None:
    f_av = eval_dir / "metrics_by_model_available.csv"
    f_co = eval_dir / "metrics_by_model_common.csv"
    if not f_av.exists() or not f_co.exists():
        return
    av = pd.read_csv(f_av); co = pd.read_csv(f_co)
    av["评价集合"] = "Available set"; co["评价集合"] = "Common set"
    df = pd.concat([av, co], ignore_index=True)
    if "spearman_mean" not in df.columns:
        return
    pivot = df.pivot_table(index="model", columns="评价集合", values="spearman_mean", aggfunc="mean")
    pivot = pivot.reindex([m for m in MODEL_ORDER if m in pivot.index])
    pivot = pivot[[c for c in ["Available set", "Common set"] if c in pivot.columns]]
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    pivot.plot(kind="bar", ax=ax, color=[SET_COLORS[c] for c in pivot.columns], edgecolor="white", linewidth=0.8)
    ax.set_xlabel("模型"); ax.set_ylabel("平均 Spearman 相关性"); ax.set_title("Available set 与 Common set 下四模型总体表现")
    ax.tick_params(axis="x", rotation=20); ax.axhline(0, linewidth=1, color="#333333"); beautify_axes(ax)
    ax.legend(title="评价集合", frameon=False)
    save_fig(fig, out_dir / "fig3_2_model_spearman_available_common", formats)

def plot_assay_spearman_distribution(eval_dir: Path, out_dir: Path, formats: List[str]) -> None:
    f = eval_dir / "metrics_by_assay_common.csv"
    if not f.exists():
        return
    df = pd.read_csv(f).dropna(subset=["spearman"]).copy()
    models = [m for m in MODEL_ORDER if m in set(df["model"])]
    data = [df.loc[df["model"] == m, "spearman"].values for m in models]
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    box = ax.boxplot(data, labels=models, showmeans=True, patch_artist=True,
                     medianprops={"color": "#222222", "linewidth": 1.2},
                     meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "#222222", "markersize": 5},
                     boxprops={"linewidth": 1.0}, whiskerprops={"linewidth": 1.0}, capprops={"linewidth": 1.0})
    for patch, model in zip(box["boxes"], models):
        patch.set_facecolor(MODEL_COLORS.get(model, "#999999")); patch.set_alpha(0.72); patch.set_edgecolor("#333333")
    for i, (vals, model) in enumerate(zip(data, models), start=1):
        if len(vals) > 0:
            ax.scatter([i]*len(vals), vals, alpha=0.65, s=24, color=MODEL_COLORS.get(model, "#999999"),
                       edgecolor="white", linewidth=0.4)
    ax.set_xlabel("模型"); ax.set_ylabel("Assay-level Spearman"); ax.set_title("Common set 下各模型在不同 assay 中的 Spearman 分布")
    ax.axhline(0, linewidth=1, color="#333333"); beautify_axes(ax)
    save_fig(fig, out_dir / "fig3_3_assay_spearman_distribution_common", formats)

def plot_virus_spearman_common(eval_dir: Path, out_dir: Path, formats: List[str]) -> None:
    f = eval_dir / "metrics_by_virus_common.csv"
    if not f.exists():
        return
    df = pd.read_csv(f)
    if "spearman_mean" not in df.columns:
        return
    pivot = df.pivot_table(index="virus", columns="model", values="spearman_mean", aggfunc="mean")
    pivot = pivot.reindex([v for v in VIRUS_ORDER if v in pivot.index])
    pivot = pivot[[m for m in MODEL_ORDER if m in pivot.columns]]
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    pivot.plot(kind="bar", ax=ax, color=[MODEL_COLORS.get(c, "#999999") for c in pivot.columns], edgecolor="white", linewidth=0.8)
    ax.set_xlabel("病毒体系"); ax.set_ylabel("平均 Spearman 相关性"); ax.set_title("不同病毒体系下四模型表现（Common set）")
    ax.tick_params(axis="x", rotation=0); ax.axhline(0, linewidth=1, color="#333333"); beautify_axes(ax)
    ax.legend(title="模型", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    save_fig(fig, out_dir / "fig3_4_virus_spearman_common", formats)

def plot_topk_metric_common(eval_dir: Path, out_dir: Path, formats: List[str], metric: str, ylabel: str, out_name: str, title: str) -> None:
    f = eval_dir / "topk_by_model_common.csv"
    if not f.exists():
        return
    df = pd.read_csv(f)
    if metric not in df.columns:
        return
    df["top_frac_label"] = (df["top_frac"] * 100).round().astype(int).astype(str) + "%"
    pivot = df.pivot_table(index="top_frac_label", columns="model", values=metric, aggfunc="mean")
    order = df[["top_frac", "top_frac_label"]].drop_duplicates().sort_values("top_frac")["top_frac_label"].tolist()
    pivot = pivot.reindex(order)
    pivot = pivot[[m for m in MODEL_ORDER if m in pivot.columns]]
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for model in pivot.columns:
        ax.plot(pivot.index, pivot[model], marker="o", linewidth=2.6, markersize=6.5, label=model,
                color=MODEL_COLORS.get(model, "#999999"))
    ax.set_xlabel("Top-K 比例"); ax.set_ylabel(ylabel); ax.set_title(title); beautify_axes(ax)
    ax.legend(title="模型", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    save_fig(fig, out_dir / out_name, formats)

def write_figure_summary(out_dir: Path) -> None:
    lines = [
        "Figures generated by 22_plot_4models_results.py",
        "=" * 70, "",
        "fig3_1_model_coverage",
        "fig3_2_model_spearman_available_common",
        "fig3_3_assay_spearman_distribution_common",
        "fig3_4_virus_spearman_common",
        "fig3_5_topk_recall_common",
        "fig3_6_topk_jaccard_common",
    ]
    (out_dir / "figure_summary.txt").write_text("\n".join(lines), encoding="utf-8")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=str, default=".")
    parser.add_argument("--eval-dir", type=str, default="data_processed/evaluation_4models")
    parser.add_argument("--out-dir", type=str, default="data_processed/evaluation_4models/figures")
    parser.add_argument("--formats", type=str, default="png,pdf")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    eval_dir = (project_root / args.eval_dir).resolve()
    out_dir = (project_root / args.out_dir).resolve()
    formats = [x.strip().lower() for x in args.formats.split(",") if x.strip()]
    if not eval_dir.exists():
        raise FileNotFoundError(f"Evaluation directory not found: {eval_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_matplotlib()
    plot_model_coverage(eval_dir, out_dir, formats)
    plot_model_spearman_available_common(eval_dir, out_dir, formats)
    plot_assay_spearman_distribution(eval_dir, out_dir, formats)
    plot_virus_spearman_common(eval_dir, out_dir, formats)
    plot_topk_metric_common(eval_dir, out_dir, formats, "topk_recall_mean", "Top-K recall", "fig3_5_topk_recall_common", "Common set 下四模型 Top-K recall 比较")
    plot_topk_metric_common(eval_dir, out_dir, formats, "topk_jaccard_mean", "Top-K Jaccard", "fig3_6_topk_jaccard_common", "Common set 下四模型 Top-K Jaccard 比较")
    write_figure_summary(out_dir)
    print("[OK] Figures generated.")

if __name__ == "__main__":
    main()
