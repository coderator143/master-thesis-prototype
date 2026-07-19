"""
Prototype runner.

Loads the manually annotated scene variables (data/scene_variables.csv),
computes an accident-risk KPI for each video, simulates one illustrative
intervention per video, and saves the results (CSVs + charts) to outputs/.

This is the "simulation harness" -- all the actual risk-modeling logic
lives in risk_model.py.
"""

import pandas as pd
import matplotlib.pyplot as plt

import risk_model

DATA_PATH = "data/scene_variables.csv"
OUTPUTS_DIR = "outputs"

# Status palette (fixed, per the project's dataviz convention): Low/Moderate/
# High risk are states, not identities, so each bar is colored by its own
# risk band rather than by "before/after".
BAND_COLORS = {
    "Low Risk": "#0ca30c",
    "Moderate Risk": "#fab219",
    "High Risk": "#d03b3b",
}
MUTED_INK = "#898781"
PRIMARY_INK = "#0b0b0b"
SECONDARY_INK = "#52514e"

# One illustrative "what if" intervention per video, chosen to change the
# scene variable most likely to have made the biggest difference. These are
# examples for demonstrating the framework, not conclusions about the real
# accidents -- see data/scene_variables.csv for the (placeholder) base
# scenario each one is applied to.
INTERVENTIONS = {
    "VRU_9": ("speed", "low", "What if the vehicle had been going slower?"),
    "VRU_10": ("weather", "clear", "What if the weather had been clear?"),
    "VRU_14": ("visibility", "high", "What if visibility had been better?"),
}


def compute_risk_table(df):
    rows = []
    for _, row in df.iterrows():
        score = risk_model.calculate_risk(
            speed=row["speed"],
            visibility=row["visibility"],
            proximity=row["proximity"],
            weather=row["weather"],
        )
        rows.append(
            {
                "video_id": row["video_id"],
                "speed": row["speed"],
                "visibility": row["visibility"],
                "proximity": row["proximity"],
                "weather": row["weather"],
                "risk_score": score,
                "risk_level": risk_model.interpret_risk(score),
            }
        )
    return pd.DataFrame(rows)


def run_interventions(df):
    rows = []
    for _, row in df.iterrows():
        video_id = row["video_id"]
        if video_id not in INTERVENTIONS:
            continue
        variable, new_value, question = INTERVENTIONS[video_id]
        base_scenario = {
            "speed": row["speed"],
            "visibility": row["visibility"],
            "proximity": row["proximity"],
            "weather": row["weather"],
        }
        result = risk_model.simulate_intervention(base_scenario, {variable: new_value})
        rows.append(
            {
                "video_id": video_id,
                "question": question,
                "variable_changed": variable,
                "original_value": base_scenario[variable],
                "new_value": new_value,
                "base_score": result["base_score"],
                "base_level": result["base_level"],
                "new_score": result["new_score"],
                "new_level": result["new_level"],
                "delta": result["delta"],
            }
        )
    return pd.DataFrame(rows)


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(MUTED_INK)
    ax.spines["bottom"].set_color(MUTED_INK)
    ax.tick_params(colors=SECONDARY_INK)
    ax.set_ylim(0, 1.0)
    ax.axhline(0.3, color=MUTED_INK, linestyle="--", linewidth=0.8, zorder=0)
    ax.axhline(0.6, color=MUTED_INK, linestyle="--", linewidth=0.8, zorder=0)


def add_bar_labels(ax, bars, scores):
    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{score:.2f}",
            ha="center",
            va="bottom",
            color=PRIMARY_INK,
            fontsize=9,
        )


def add_band_legend(ax):
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=color) for color in BAND_COLORS.values()
    ]
    # Placed below the axes (not "upper right") so it never collides with a
    # bar label near the top of the 0-1 range, regardless of bar height.
    ax.legend(
        handles,
        BAND_COLORS.keys(),
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=3,
        fontsize=8,
    )


def plot_intervention_chart(intervention_row, out_path):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    labels = ["Before", "After"]
    scores = [intervention_row["base_score"], intervention_row["new_score"]]
    levels = [intervention_row["base_level"], intervention_row["new_level"]]
    colors = [BAND_COLORS[level] for level in levels]

    bars = ax.bar(labels, scores, color=colors, width=0.5)
    add_bar_labels(ax, bars, scores)
    style_axes(ax)
    add_band_legend(ax)

    ax.set_ylabel("Risk score (0-1)", color=SECONDARY_INK)
    ax.set_title(intervention_row["question"], color=PRIMARY_INK, fontsize=10, wrap=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_summary_chart(risk_df, out_path):
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = [BAND_COLORS[level] for level in risk_df["risk_level"]]

    bars = ax.bar(risk_df["video_id"], risk_df["risk_score"], color=colors, width=0.5)
    add_bar_labels(ax, bars, risk_df["risk_score"])
    style_axes(ax)
    add_band_legend(ax)

    ax.set_ylabel("Risk score (0-1)", color=SECONDARY_INK)
    ax.set_title("Estimated accident risk per video", color=PRIMARY_INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    df = pd.read_csv(DATA_PATH)

    risk_df = compute_risk_table(df)
    print("Accident-risk KPI per video:\n")
    print(risk_df.to_string(index=False))
    risk_df.to_csv(f"{OUTPUTS_DIR}/risk_results.csv", index=False)

    intervention_df = run_interventions(df)
    print("\nIntervention simulation results:\n")
    print(intervention_df.to_string(index=False))
    intervention_df.to_csv(f"{OUTPUTS_DIR}/intervention_results.csv", index=False)

    plot_summary_chart(risk_df, f"{OUTPUTS_DIR}/all_videos_risk_summary.png")
    for _, row in intervention_df.iterrows():
        plot_intervention_chart(row, f"{OUTPUTS_DIR}/{row['video_id']}_intervention.png")

    print(f"\nSaved CSVs and charts to {OUTPUTS_DIR}/")


if __name__ == "__main__":
    main()
