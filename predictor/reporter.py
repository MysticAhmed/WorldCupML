"""ReportGenerator for the FIFA World Cup Predictor."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix as sk_confusion_matrix

from predictor.simulator import SimulationResult


class ReportGenerator:
    """Produces charts and summary CSV from simulation and model results."""

    def __init__(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir

    def bar_chart(self, sim_result: SimulationResult) -> None:
        """Horizontal bar chart of top 10 teams by win probability."""
        win_probs = sim_result.win_probabilities
        top10 = sorted(win_probs.items(), key=lambda x: x[1], reverse=True)[:10]
        teams = [t for t, _ in top10]
        probs = [p for _, p in top10]

        fig, ax = plt.subplots()
        ax.barh(teams[::-1], probs[::-1])
        ax.set_xlabel("Win Probability")
        ax.set_title("Top 10 Teams by Tournament Win Probability")
        plt.tight_layout()
        fig.savefig(os.path.join(self.output_dir, "win_probability_bar_chart.png"))
        plt.close(fig)

    def confusion_matrix(self, y_true, y_pred) -> None:
        """Confusion matrix heatmap saved as confusion_matrix.png."""
        cm = sk_confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        plt.tight_layout()
        fig.savefig(os.path.join(self.output_dir, "confusion_matrix.png"))
        plt.close(fig)

    def feature_importance(self, importance_series: pd.Series) -> None:
        """Horizontal bar chart of top 20 features by importance."""
        top20 = importance_series.nlargest(20)

        fig, ax = plt.subplots()
        ax.barh(top20.index[::-1], top20.values[::-1])
        ax.set_xlabel("Importance")
        ax.set_title("Top 20 Feature Importances")
        plt.tight_layout()
        fig.savefig(os.path.join(self.output_dir, "feature_importance.png"))
        plt.close(fig)

    def summary_csv(self, sim_result: SimulationResult) -> None:
        """Write summary CSV with team win/semifinal/final probabilities."""
        rows = [
            {
                "team": team,
                "win_prob": sim_result.win_probabilities[team],
                "semifinal_prob": sim_result.semifinal_probabilities[team],
                "final_prob": sim_result.final_probabilities[team],
            }
            for team in sim_result.win_probabilities
        ]
        df = pd.DataFrame(rows).sort_values("win_prob", ascending=False)
        df.to_csv(os.path.join(self.output_dir, "simulation_summary.csv"), index=False)
