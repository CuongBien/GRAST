"""Standalone SFT research curves, rebuilt from checkpoint-owned history."""

import csv
import io

from llm_scratch.training.pretrain import atomic_text
from llm_scratch.utils.io import write_json


def export_sft_curves(history, directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "history.json", history)
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(["series", "step", "attempt", "loss", "loss_tokens", "skipped"])
    for series, rows in history.items():
        for r in rows:
            writer.writerow(
                [
                    series,
                    r["step"],
                    r["attempt"],
                    r["loss"],
                    r["loss_tokens"],
                    r.get("skipped", False),
                ]
            )
    atomic_text(directory / "losses.csv", output.getvalue())
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    try:
        for series, rows in history.items():
            selected = [r for r in rows if not r.get("skipped", False)]
            if selected:
                ax.plot(
                    [r["step"] for r in selected],
                    [r["loss"] for r in selected],
                    label=series,
                    marker="o" if series == "validation" else None,
                )
        ax.set(
            xlabel="Successful SFT optimizer updates",
            ylabel="Target-token cross entropy",
            title="CoT SFT loss (prompt and padding masked)",
        )
        ax.grid(alpha=0.25)
        ax.legend()
        for suffix in ("png", "svg"):
            fig.savefig(directory / f"loss_curves.{suffix}", dpi=160)
    finally:
        plt.close(fig)
