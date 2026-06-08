import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_loss(loss_history_path: str, output: str):
    path = Path(loss_history_path)
    if not path.exists():
        print(f"Error: {path} not found. Run training first to generate loss_history.json.")
        return

    with open(path) as f:
        data = json.load(f)

    train = data["train"]
    eval_data = data["eval"]

    train_steps = [p["step"] for p in train]
    train_losses = [p["loss"] for p in train]
    eval_steps = [p["step"] for p in eval_data]
    eval_losses = [p["eval_loss"] for p in eval_data]

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(train_steps, train_losses, "b-", linewidth=1.5, label="Train Loss", marker="o", markersize=3, alpha=0.7)
    ax.scatter(eval_steps, eval_losses, c="red", s=40, zorder=5, label="Eval Loss", edgecolors="darkred")
    ax.plot(eval_steps, eval_losses, "r--", linewidth=1.2, alpha=0.5)

    if eval_losses:
        ax.annotate(f"{eval_losses[0]:.2f}", (eval_steps[0], eval_losses[0]),
                    textcoords="offset points", xytext=(-15, -15),
                    ha="right", fontsize=9, color="darkred")
        ax.annotate(f"{eval_losses[-1]:.2f}", (eval_steps[-1], eval_losses[-1]),
                    textcoords="offset points", xytext=(5, 12),
                    ha="center", fontsize=9, color="darkred")

    min_train = min(train_losses)
    ax.axhline(y=min_train, color="blue", linestyle="--", alpha=0.2, linewidth=0.8)

    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title(f"Qwen2.5-7B LoRA Fine-tuning Loss\n"
                 f"Train: {len(train)} points, {train[0]['step']}-{train[-1]['step']} steps | "
                 f"Train loss: {train_losses[0]:.2f} → {train_losses[-1]:.3f}\n"
                 f"Eval loss: {eval_losses[0]:.2f} → {eval_losses[-1]:.3f}" if eval_losses else "",
                 fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()
    print(f"Plot saved to: {output}")
    print(f"Train: {len(train)} points, loss {train_losses[0]:.4f} → {train_losses[-1]:.4f}")
    print(f"Eval:  {len(eval_data)} points, loss {eval_losses[0]:.4f} → {eval_losses[-1]:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training loss curves from loss_history.json")
    parser.add_argument("--loss_history", default="./outputs/qwen2.5-7b-lora-output/loss_history.json")
    parser.add_argument("--output", default="./outputs/loss_curve.png")
    args = parser.parse_args()
    plot_loss(args.loss_history, args.output)
