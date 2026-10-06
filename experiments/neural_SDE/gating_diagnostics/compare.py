import json
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('default')
sns.set_theme(style="whitegrid")


def load_run(variant_dir: Path):
    variant_dir = Path(variant_dir)
    try:
        with open(variant_dir / "variant.json") as f:
            meta = json.load(f)
    except FileNotFoundError:
        meta = {"name": variant_dir.name, "overrides": {}}

    try:
        metrics = pd.read_csv(variant_dir / "epoch_metrics.csv")
    except FileNotFoundError:
        metrics = None

    try:
        with open(variant_dir / "val_metrics.json") as f:
            val_metrics = json.load(f)
    except FileNotFoundError:
        val_metrics = {}

    try:
        with open(variant_dir / "manifest.json") as f:
            manifest = json.load(f)
    except FileNotFoundError:
        manifest = {}

    return {
        "name": meta["name"],
        "overrides": meta["overrides"],
        "metrics": metrics,
        "val_metrics": val_metrics,
        "manifest": manifest,
        "dir": variant_dir,
    }


def summary_table(runs):
    records = []
    for r in runs:
        vm = r["val_metrics"]
        records.append(
            {
                "Variant": r["name"],
                "Final Loss": vm[0].get("val/loss", None)
                if isinstance(vm, list) and vm
                else (vm.get("val/loss") if isinstance(vm, dict) else None),
                "SDE Loss": vm[0].get("val/loss_sde", None)
                if isinstance(vm, list) and vm
                else (vm.get("val/loss_sde") if isinstance(vm, dict) else None),
                "Reg Term": vm[0].get("val/loss_gate_reg_term", None)
                if isinstance(vm, list) and vm
                else (
                    vm.get("val/loss_gate_reg_term") if isinstance(vm, dict) else None
                ),
                "Entropy (Last Ep)": r["metrics"]
                .query("split == 'validation'")
                .iloc[-1]["entropy"]
                if r["metrics"] is not None and not r["metrics"].empty
                else None,
                "Winner Share": r["metrics"]
                .query("split == 'validation'")
                .iloc[-1]["p_bounce_winner_share"]
                if r["metrics"] is not None and not r["metrics"].empty
                else None,
            }
        )
    return pd.DataFrame(records)


def plot_utilization(runs):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    experts = ["p_bounce", "p_break", "p_hover"]
    titles = ["Bounce Expert", "Break Expert", "Hover Expert"]
    
    for ax, expert, title in zip(axes, experts, titles):
        for r in runs:
            m = r["metrics"]
            if m is not None:
                val_m = m[m["split"] == "validation"]
                if expert in val_m.columns:
                    sns.lineplot(data=val_m, x="epoch", y=expert, label=r["name"], ax=ax)
        
        ax.set_title(title)
        ax.set_ylabel("Mean Assignment Probability")
        ax.set_ylim(0, 1)
        ax.grid(True)
        ax.legend()
        
    plt.tight_layout()
    from IPython.display import display
    display(plt.gcf())
    plt.close()


def plot_distance(runs):
    plt.figure(figsize=(10, 6))
    for r in runs:
        m = r["metrics"]
        if m is not None:
            val_m = m[m["split"] == "validation"]
            if "entropy" in val_m.columns:
                sns.lineplot(data=val_m, x="epoch", y="entropy", label=r["name"])
    plt.title("Routing Entropy (Sharpness) over Epochs")
    plt.ylabel("Entropy")
    plt.legend()
    plt.grid(True)
    from IPython.display import display

    display(plt.gcf())
    plt.close()


def figure_grid(runs, filename="event_heatmaps.png"):
    n = len(runs)
    if n == 0:
        return
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, r in zip(axes, runs):
        img_path = r["dir"] / filename
        if img_path.exists():
            img = plt.imread(img_path)
            ax.imshow(img)
            ax.axis("off")
        ax.set_title(r["name"])
    plt.tight_layout()
    from IPython.display import display

    display(plt.gcf())
    plt.close()
