import matplotlib.pyplot as plt
import os
import glob
from tensorboard.backend.event_processing import event_accumulator


def main():
    log_dir = "workspace/neural_SDE/logs/train_neural_SDE"
    # Find the most recent run directory
    version_dirs = sorted(
        [
            os.path.join(log_dir, d)
            for d in os.listdir(log_dir)
            if os.path.isdir(os.path.join(log_dir, d))
        ],
        key=os.path.getmtime,
        reverse=True,
    )
    if not version_dirs:
        print("No log directories found.")
        return

    latest_dir = version_dirs[0]

    # Since multiple runs are saving to the SAME directory ("US_Stocks"),
    # pointing EventAccumulator at the whole directory merges all historical runs together.
    # To get only the latest run's data, we specifically load the newest `.0` file.
    event_files = sorted(
        glob.glob(os.path.join(latest_dir, "events.out.tfevents.*.0")),
        key=os.path.getmtime,
        reverse=True,
    )

    if event_files:
        event_path = event_files[0]
        print(f"Reading latest run event file: {os.path.basename(event_path)}")
    else:
        event_path = latest_dir
        print(f"Reading logs from directory: {latest_dir}")

    ea = event_accumulator.EventAccumulator(event_path)
    ea.Reload()

    keys_train = ["train/pi_mean_e0", "train/pi_mean_e1", "train/pi_mean_e2"]
    keys_val = ["val/pi_mean_e0", "val/pi_mean_e1", "val/pi_mean_e2"]
    labels = ["Bounce", "Break", "Hover"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # standard matplotlib colors

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    fig.suptitle("1. Expert utilization through training", fontsize=16)

    # Derive n windows/epoch
    import re

    batch_size = 256
    hparams_path = os.path.join(latest_dir, "hparams.yaml")
    if os.path.exists(hparams_path):
        with open(hparams_path, "r") as f:
            m = re.search(r"batch_size:\s*(\d+)", f.read())
            if m:
                batch_size = int(m.group(1))

    steps_per_epoch = 1000  # default fallback
    if "epoch" in ea.scalars.Keys():
        epoch_events = ea.scalars.Items("epoch")
        epochs = [e.value for e in epoch_events]
        steps = [e.step for e in epoch_events]
        if max(epochs) > 0:
            steps_per_epoch = max(steps) / max(epochs)

    train_windows = int(steps_per_epoch * batch_size)
    val_windows = int(train_windows * 0.25)  # assuming 80/20 split

    axes[0].set_title(f"Train | n={train_windows:,} windows/epoch", fontsize=11)
    axes[1].set_title(f"Validation | n={val_windows:,} windows/epoch", fontsize=11)

    for ax_idx, (ax, keys) in enumerate(zip(axes, [keys_train, keys_val])):
        for i, key in enumerate(keys):
            if key in ea.scalars.Keys():
                events = ea.scalars.Items(key)
                # Since it's logged per epoch, we can just use 0-based epoch indices
                epochs = list(range(len(events)))
                vals = [e.value for e in events]
                ax.plot(epochs, vals, marker=".", label=labels[i], color=colors[i])
            else:
                print(f"Key {key} not found in logs.")

        ax.set_xlabel("Epoch (zero-based)")
        if ax_idx == 0:
            ax.set_ylabel("Gate probability")
        else:
            ax.set_ylabel(
                "Gate probability"
            )  # Although sharey is true, setting ylabel on right can mimic the image

        ax.set_ylim(0, 1.0)
        ax.axhline(1 / 3, color="gray", linestyle="dotted", alpha=0.7)
        ax.legend()
        ax.grid(True, alpha=0.3)
        # Remove top and right spines to match the style
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    caption = (
        "Sample-weighted mean gates. Training averages span changing weights; validation uses the same windows each epoch.\n"
        "The one-third line is a reference, not a required target. Unequal use alone does not establish collapse."
    )
    plt.figtext(
        0.5,
        0.01,
        caption,
        wrap=True,
        horizontalalignment="center",
        fontsize=9,
        color="gray",
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, top=0.88)

    out_path = "expert_utilization_reproduced.png"
    plt.savefig(out_path, dpi=300)
    print(f"Saved visualization to {out_path}")


if __name__ == "__main__":
    main()
