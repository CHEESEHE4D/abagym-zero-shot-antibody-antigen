#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
25_plot_interface_results.py

界面/非界面结果作图脚本（高辨识度配色版）
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Iterable, List, Optional
import pandas as pd
import matplotlib.pyplot as plt

MODEL_ORDER = ["ESM1v", "ESM2", "SaProt", "ESM-IF-AF2"]
GROUP_ORDER = ["interface", "non_interface"]

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

def find_csv(folder: Path, base_name: str) -> Optional[Path]:
    for p in [folder / base_name, folder / f"simulated_{base_name}"]:
        if p.exists():
            return p
    return None

def normalize_group_label(x: str) -> str:
    if pd.isna(x):
        return ""
    x = str(x).strip()
    if x in ["interface", "界面"]:
        return "interface"
    if x in ["non_interface", "non-interface", "noninterface", "非界面"]:
        return "non_interface"
    return x

def group_display(x: str) -> str:
    return {"interface": "界面", "non_interface": "非界面"}.get(x, x)

def plot_interface_spearman_common(in_dir: Path, out_dir: Path, formats: List[str]) -> None:
    p = find_csv(in_dir, "metrics_by_interface_common.csv")
    if p is None:
        return
    df = pd.read_csv(p)
    df["interface_group"] = df["interface_group"].map(normalize_group_label)
    pivot = df.pivot_table(index="interface_group", columns="model", values="spearman_mean", aggfunc="mean")
    pivot = pivot.reindex([g for g in GROUP_ORDER if g in pivot.index])
    pivot = pivot[[m for m in MODEL_ORDER if m in pivot.columns]]
    pivot.index = [group_display(x) for x in pivot.index]
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    pivot.plot(kind="bar", ax=ax, color=[MODEL_COLORS.get(c, "#999999") for c in pivot.columns], edgecolor="white", linewidth=0.8)
    ax.set_xlabel("突变分组"); ax.set_ylabel("平均 Spearman 相关性"); ax.set_title("界面与非界面突变中的模型表现（Common set）")
    ax.tick_params(axis="x", rotation=0); ax.axhline(0, color="#333333", linewidth=1); beautify_axes(ax)
    ax.legend(title="模型", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    save_fig(fig, out_dir / "fig_interface_1_spearman_common", formats)

def plot_interface_available_common(in_dir: Path, out_dir: Path, formats: List[str]) -> None:
    p_av = find_csv(in_dir, "metrics_by_interface_available.csv")
    p_co = find_csv(in_dir, "metrics_by_interface_common.csv")
    if p_av is None or p_co is None:
        return
    av = pd.read_csv(p_av); co = pd.read_csv(p_co)
    av["评价集合"] = "Available set"; co["评价集合"] = "Common set"
    df = pd.concat([av, co], ignore_index=True)
    df["interface_group"] = df["interface_group"].map(normalize_group_label)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), sharey=True)
    for ax, group in zip(axes, GROUP_ORDER):
        sub = df[df["interface_group"] == group].copy()
        pivot = sub.pivot_table(index="model", columns="评价集合", values="spearman_mean", aggfunc="mean")
        pivot = pivot.reindex([m for m in MODEL_ORDER if m in pivot.index])
        pivot = pivot[[c for c in ["Available set", "Common set"] if c in pivot.columns]]
        pivot.plot(kind="bar", ax=ax, color=[SET_COLORS.get(c, "#999999") for c in pivot.columns], edgecolor="white", linewidth=0.8)
        ax.set_title(group_display(group)); ax.set_xlabel("模型"); ax.tick_params(axis="x", rotation=25)
        ax.axhline(0, color="#333333", linewidth=1); beautify_axes(ax); ax.legend(title="评价集合", frameon=False)
    axes[0].set_ylabel("平均 Spearman 相关性")
    fig.suptitle("Available set 与 Common set 下界面分组表现", y=1.03)
    save_fig(fig, out_dir / "fig_interface_2_available_common", formats)

def plot_assay_distribution_common(in_dir: Path, out_dir: Path, formats: List[str]) -> None:
    p = find_csv(in_dir, "metrics_by_assay_interface_common.csv")
    if p is None:
        return
    df = pd.read_csv(p).dropna(subset=["spearman"]).copy()
    df["interface_group"] = df["interface_group"].map(normalize_group_label)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), sharey=True)
    for ax, group in zip(axes, GROUP_ORDER):
        sub = df[df["interface_group"] == group]
        models = [m for m in MODEL_ORDER if m in set(sub["model"])]
        data = [sub.loc[sub["model"] == m, "spearman"].values for m in models]
        box = ax.boxplot(data, labels=models, showmeans=True, patch_artist=True,
                         medianprops={"color": "#222222", "linewidth": 1.2},
                         meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "#222222", "markersize": 5},
                         boxprops={"linewidth": 1.0}, whiskerprops={"linewidth": 1.0}, capprops={"linewidth": 1.0})
        for patch, model in zip(box["boxes"], models):
            patch.set_facecolor(MODEL_COLORS.get(model, "#999999")); patch.set_alpha(0.72); patch.set_edgecolor("#333333")
        for i, (vals, model) in enumerate(zip(data, models), start=1):
            if len(vals) > 0:
                ax.scatter([i]*len(vals), vals, alpha=0.65, s=20, color=MODEL_COLORS.get(model, "#999999"),
                           edgecolor="white", linewidth=0.4)
        ax.set_title(group_display(group)); ax.set_xlabel("模型"); ax.axhline(0, color="#333333", linewidth=1); beautify_axes(ax)
    axes[0].set_ylabel("Assay-level Spearman")
    fig.suptitle("界面与非界面分组下 assay-level Spearman 分布（Common set）", y=1.03)
    save_fig(fig, out_dir / "fig_interface_3_assay_distribution_common", formats)

def plot_topk_recall_common(in_dir: Path, out_dir: Path, formats: List[str]) -> None:
    p = find_csv(in_dir, "topk_by_interface_common.csv")
    if p is None:
        return
    df = pd.read_csv(p)
    df["interface_group"] = df["interface_group"].map(normalize_group_label)
    if "topk_recall_mean" not in df.columns:
        return
    df["top_frac_label"] = (df["top_frac"] * 100).round().astype(int).astype(str) + "%"
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), sharey=True)
    for ax, group in zip(axes, GROUP_ORDER):
        sub = df[df["interface_group"] == group].copy()
        pivot = sub.pivot_table(index="top_frac_label", columns="model", values="topk_recall_mean", aggfunc="mean")
        order = sub[["top_frac", "top_frac_label"]].drop_duplicates().sort_values("top_frac")["top_frac_label"].tolist()
        pivot = pivot.reindex(order)
        pivot = pivot[[m for m in MODEL_ORDER if m in pivot.columns]]
        for model in pivot.columns:
            ax.plot(pivot.index, pivot[model], marker="o", linewidth=2.6, markersize=6.5, label=model,
                    color=MODEL_COLORS.get(model, "#999999"))
        ax.set_title(group_display(group)); ax.set_xlabel("Top-K 比例"); beautify_axes(ax); ax.legend(title="模型", frameon=False)
    axes[0].set_ylabel("Top-K recall")
    fig.suptitle("界面与非界面分组下 Top-K recall 比较（Common set）", y=1.03)
    save_fig(fig, out_dir / "fig_interface_4_topk_recall_common", formats)

def plot_coverage_by_interface(in_dir: Path, out_dir: Path, formats: List[str]) -> None:
    p = find_csv(in_dir, "model_score_coverage_by_interface.csv")
    if p is None:
        return
    df = pd.read_csv(p)
    df["interface_group"] = df["interface_group"].map(normalize_group_label)
    y_col = "n_usable_rows" if "n_usable_rows" in df.columns else "n_rows"
    pivot = df.pivot_table(index="interface_group", columns="model", values=y_col, aggfunc="sum")
    pivot = pivot.reindex([g for g in GROUP_ORDER if g in pivot.index])
    pivot = pivot[[m for m in MODEL_ORDER if m in pivot.columns]]
    pivot.index = [group_display(x) for x in pivot.index]
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    pivot.plot(kind="bar", ax=ax, color=[MODEL_COLORS.get(c, "#999999") for c in pivot.columns], edgecolor="white", linewidth=0.8)
    ax.set_xlabel("突变分组"); ax.set_ylabel("有效突变记录数"); ax.set_title("界面与非界面分组下模型覆盖情况")
    ax.tick_params(axis="x", rotation=0); beautify_axes(ax)
    ax.legend(title="模型", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    save_fig(fig, out_dir / "fig_interface_5_coverage_by_interface", formats)

def write_summary(out_dir: Path) -> None:
    lines = [
        "Interface figures generated by 25_plot_interface_results.py",
        "=" * 70, "",
        "fig_interface_1_spearman_common",
        "fig_interface_2_available_common",
        "fig_interface_3_assay_distribution_common",
        "fig_interface_4_topk_recall_common",
        "fig_interface_5_coverage_by_interface",
    ]
    (out_dir / "figure_interface_summary.txt").write_text("\n".join(lines), encoding="utf-8")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=str, default=".")
    parser.add_argument("--interface-dir", type=str, default="data_processed/simulation_interface_supplement")
    parser.add_argument("--out-dir", type=str, default="")
    parser.add_argument("--formats", type=str, default="png,pdf")
    args = parser.parse_args()
    project_root = Path(args.project_root).resolve()
    in_dir = (project_root / args.interface_dir).resolve()
    if args.out_dir:
        out_dir = (project_root / args.out_dir).resolve()
    else:
        out_dir = in_dir / "figures_interface"
    formats = [x.strip().lower() for x in args.formats.split(",") if x.strip()]
    if not in_dir.exists():
        raise FileNotFoundError(f"Interface result directory not found: {in_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_matplotlib()
    plot_interface_spearman_common(in_dir, out_dir, formats)
    plot_interface_available_common(in_dir, out_dir, formats)
    plot_assay_distribution_common(in_dir, out_dir, formats)
    plot_topk_recall_common(in_dir, out_dir, formats)
    plot_coverage_by_interface(in_dir, out_dir, formats)
    write_summary(out_dir)
    print("[OK] Interface figures generated.")

if __name__ == "__main__":
    main()
