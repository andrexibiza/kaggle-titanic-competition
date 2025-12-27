# Kaggle Titanic Competition: The "Small Data" Solution

![Status](https://img.shields.io/badge/Status-Complete-success)
![Score](https://img.shields.io/badge/Best%20Score-0.78947-brightgreen)
![Tech](https://img.shields.io/badge/R-Caret%20%7C%20XGBoost%20%7C%20Ranger-blue)

An exhaustive research project optimizing predictive accuracy for the Kaggle Titanic dataset ($N_{train}=891$).

## 🏆 Champion Result (V4)

* **Leaderboard Score**: `0.78947` (Top ~18%)
* **Architecture**: Weighted Soft Voting Ensemble
* **Components**: XGBoost (Non-linear) + Random Forest (Variance reduction) + GLMnet (Linear baseline)

> **Key Insight**: On small tabular datasets, **Variance is the enemy**. Complex methods like Deep Learning, Stacking, and Pseudo-Labeling consistently underperformed the simpler, robust V4 ensemble.

## 📂 Core Files

| File | Description |
| :--- | :--- |
| **[`titanic_champion_solution.Rmd`](titanic_champion_solution.Rmd)** | **Start Here**. A fully documented R Markdown notebook explaining the winning strategy, feature engineering, and model architecture. |
| [`titanic_progress_report.md`](titanic_progress_report.md) | A comprehensive research log detailing all 13 iterations, including failed experiments with Deep Learning and Stacking. |
| `titanic_v4.R` | The raw R script for the champion model. |
| `submission_v4.csv` | The submission file achieving 0.78947. |

## 🧪 Research Journey (Summary)

We tested 13 different strategies to break the 0.80 barrier.

| Strategy | Score | Outcome |
| :--- | :--- | :--- |
| **V4 (Simple Ensemble)** | **0.789** | **Champion**. Best trade-off of Bias/Variance. |
| V11 (Seed Averaging) | 0.787 | Stable, but smoothed out peak performance. |
| V6 (Deep Learning) | 0.775 | Failed. Too data-hungry for 891 rows. |
| V9 (Stacking) | 0.772 | Failed. Meta-learner couldn't extract stable signal. |
| V13 (Surgical Vote) | 0.784 | Failed. "Logic" overrides proved brittle. |

## 🚀 Usage

### Prerequisites

* R (4.0+)
* Packages: `caret`, `dplyr`, `stringr`, `xgboost`, `ranger`, `glmnet`, `knitr`

### Replication

1. Clone the repository.
2. Open `titanic_champion_solution.Rmd` in RStudio or VS Code.
3. Run all chunks to generate the analysis and `submission_champion_v4.csv`.

```r
# Or run from command line
Rscript titanic_v4.R
```

## 👥 Authors

* **AndrexIbiza**
* **Antigravity** (AI Pair Programmer)
