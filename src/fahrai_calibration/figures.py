"""Draw the manuscript figures from the architecture and saved summaries."""

from pathlib import Path


def draw_architecture(output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, Rectangle
    from matplotlib.path import Path as MPath

    output.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 3.15))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})

    def box(x, y, w, h, title, body, fs=7.6):
        ax.add_patch(Rectangle((x, y), w, h, facecolor="white", edgecolor=".2", lw=0.8))
        ax.text(
            x + w / 2, y + h - 0.034, title, ha="center", va="top", fontsize=8.2, fontweight="bold"
        )
        ax.text(x + w / 2, y + 0.028, body, ha="center", va="bottom", fontsize=fs, linespacing=1.15)

    def arrow(a, b, dashed=False):
        ax.add_patch(
            FancyArrowPatch(
                a,
                b,
                arrowstyle="-|>",
                mutation_scale=9,
                color=".2",
                lw=0.8,
                linestyle="--" if dashed else "-",
            )
        )

    def routed(points):
        path = MPath(points, [MPath.MOVETO] + [MPath.LINETO] * (len(points) - 1))
        ax.add_patch(
            FancyArrowPatch(
                path=path, arrowstyle="-|>", mutation_scale=9, color=".2", lw=0.8, linestyle="--"
            )
        )

    box(
        0.075,
        0.72,
        0.365,
        0.25,
        "State Analyzer",
        "Ability, retention, evidence\nState and raw score",
    )
    box(
        0.55,
        0.72,
        0.365,
        0.25,
        "Task selection",
        "Task Selector · Repetition Scheduler\n"
        "Personalization Engine\nCandidate tasks and selector versions",
        7.1,
    )
    box(0.075, 0.405, 0.365, 0.215, "Assessment", "Progress Evaluator\nReadiness Predictor")
    box(
        0.55,
        0.405,
        0.365,
        0.215,
        "Learner interaction",
        "Selected task and answer\nEvidence for the next step",
    )
    box(
        0.075,
        0.025,
        0.84,
        0.23,
        "Recalibration audit for a fixed target selector",
        "Check support → fit weighted correction → evaluate on held-out data\n"
        "Return correction, target-policy version and reliability diagnostics",
        7.5,
    )
    arrow((0.44, 0.835), (0.55, 0.835))
    arrow((0.26, 0.72), (0.26, 0.62))
    arrow((0.735, 0.72), (0.735, 0.62))
    arrow((0.55, 0.59), (0.44, 0.735))
    routed([(0.075, 0.83), (0.025, 0.83), (0.025, 0.29), (0.16, 0.29), (0.16, 0.255)])
    ax.text(0.034, 0.545, "State, score", rotation=90, ha="left", va="center", fontsize=7.2)
    routed([(0.915, 0.83), (0.975, 0.83), (0.975, 0.29), (0.86, 0.29), (0.86, 0.255)])
    ax.text(
        0.961,
        0.545,
        "Historical and target\nselection probabilities",
        rotation=90,
        ha="right",
        va="center",
        fontsize=7.2,
    )
    arrow((0.66, 0.405), (0.66, 0.255), True)
    ax.text(0.68, 0.326, "Task, answer", ha="left", va="center", fontsize=7.2)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(output / "Kholodniak_O_O-pictures-fig_1.jpg", dpi=300)
    plt.close(fig)


def draw_overlap(results: dict, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})

    res = results
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.8), layout="constrained")
    styles = [
        ("Raw", "o", "-", ".65"),
        ("Global", "s", "--", ".35"),
        ("Regime", "^", ":", ".1"),
        ("Weighted global", "D", "-.", ".5"),
        ("Weighted regime", "v", "-", ".0"),
        ("Context cells", "x", "--", ".0"),
    ]
    for method, marker, ls, color in styles:
        ss = [s for s in res["summary"] if s["method"] == method]
        for ax, key in zip(axes, ["ece", "excess_brier"]):
            ax.errorbar(
                [0.2, 0.05, 0.01],
                [s[key]["mean"] for s in ss],
                yerr=[1.96 * s[key]["mcse"] for s in ss],
                marker=marker,
                ls=ls,
                ms=4,
                label=method,
                color=color,
                capsize=2,
                lw=1,
            )
    for ax, title in zip(axes, ["Calibration error (15 bins)", "Excess Brier score"]):
        ax.set_xscale("log")
        ax.invert_xaxis()
        ax.set_xticks([0.2, 0.05, 0.01], ["0.20", "0.05", "0.01"])
        ax.set_xlabel("Logging probability of harder action", fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.2)
        ax.tick_params(labelsize=8)
    axes[0].legend(fontsize=6.5, loc="best", framealpha=0.8)
    fig.savefig(output / "Kholodniak_O_O-pictures-fig_2.jpg", dpi=300)
    plt.close(fig)
