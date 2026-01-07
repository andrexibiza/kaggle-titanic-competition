# SPICY: Self-Perfecting through Iterative Critique Yielding

## The Adversarial Agentic Dynamic for AI Quality Improvement

This document describes the BDSM (Beneficial Dialectic for Self-Mastery) agentic framework used to improve the Titanic portfolio notebook through adversarial critique and remediation.

---

## Theoretical Foundation

The SPICY framework implements a **Dominant-Submissive Agent Architecture** where:

1. **The Critic Agent (Dominant)**: Assumes an expert persona and ruthlessly identifies flaws
2. **The Implementer Agent (Submissive)**: Accepts criticism without ego defense and executes fixes
3. **The Overlord (Human)**: Orchestrates the dynamic and provides final judgment

This architecture exploits the psychological principle that **humans defend their work against criticism**, but AI agents can be instructed to fully accept criticism and improve without ego interference.

---

## Implementation

### Phase 1: Summon the Critic

```python
"""
SPICY Framework: Adversarial Agentic Quality Improvement
"""

from dataclasses import dataclass
from typing import List, Callable
from enum import Enum

class CritiqueCategory(Enum):
    STATISTICAL = "statistical_methodology"
    CODE_QUALITY = "code_quality"
    PRESENTATION = "presentation"
    METHODOLOGY = "methodology"
    ACCESSIBILITY = "accessibility"

@dataclass
class Critique:
    id: int
    category: CritiqueCategory
    severity: str  # "critical", "major", "minor"
    description: str
    fix_strategy: str
    affected_cells: List[str]

class CriticAgent:
    """
    The Dominant: Assumes expert persona and identifies flaws ruthlessly.
    No mercy. No excuses. Only truth.
    """

    def __init__(self, persona: str = "Kaggle Grandmaster"):
        self.persona = persona
        self.critiques: List[Critique] = []

    def analyze(self, notebook_content: str) -> List[Critique]:
        """
        Analyze notebook and generate critiques.
        The Critic shows no mercy.
        """
        critiques = []

        # Statistical critiques
        critiques.append(Critique(
            id=1,
            category=CritiqueCategory.STATISTICAL,
            severity="critical",
            description="Correlation analysis on n=8 points lacks statistical significance",
            fix_strategy="Add bootstrap/permutation test with p-value",
            affected_cells=["cell-13"]
        ))

        critiques.append(Critique(
            id=2,
            category=CritiqueCategory.STATISTICAL,
            severity="major",
            description="Bias-variance decomposition cited is for regression, not classification",
            fix_strategy="Add classification-specific framework (Brier score decomposition)",
            affected_cells=["cell-1"]
        ))

        critiques.append(Critique(
            id=3,
            category=CritiqueCategory.STATISTICAL,
            severity="major",
            description="No confidence intervals on reported scores",
            fix_strategy="Add Wilson score intervals or bootstrap CIs",
            affected_cells=["cell-0", "cell-12"]
        ))

        critiques.append(Critique(
            id=4,
            category=CritiqueCategory.METHODOLOGY,
            severity="major",
            description="$5 fare filter is arbitrary without justification",
            fix_strategy="Add sensitivity analysis showing robustness across thresholds",
            affected_cells=["cell-15"]
        ))

        # Methodological critiques
        critiques.append(Critique(
            id=5,
            category=CritiqueCategory.METHODOLOGY,
            severity="major",
            description="Using leaderboard as validation set (test set exploitation)",
            fix_strategy="Acknowledge limitation explicitly in methodology section",
            affected_cells=["cell-5"]
        ))

        critiques.append(Critique(
            id=6,
            category=CritiqueCategory.METHODOLOGY,
            severity="critical",
            description="Conservative adjustment is leaderboard probing (p-hacking)",
            fix_strategy="Frame as exploratory finding with appropriate caveats",
            affected_cells=["cell-12", "cell-22"]
        ))

        critiques.append(Critique(
            id=7,
            category=CritiqueCategory.METHODOLOGY,
            severity="major",
            description="Missing comprehensive EDA section",
            fix_strategy="Add correlation matrix, feature distributions, survival rates",
            affected_cells=["new cell after cell-4"]
        ))

        critiques.append(Critique(
            id=8,
            category=CritiqueCategory.METHODOLOGY,
            severity="major",
            description="No feature importance analysis (SHAP, permutation)",
            fix_strategy="Add permutation importance and partial dependence plots",
            affected_cells=["new cell after cell-16"]
        ))

        critiques.append(Critique(
            id=9,
            category=CritiqueCategory.METHODOLOGY,
            severity="minor",
            description="Train-test shift hypothesis is unfalsifiable",
            fix_strategy="Acknowledge explicitly as limitation",
            affected_cells=["cell-22"]
        ))

        # Code quality critiques
        critiques.append(Critique(
            id=10,
            category=CritiqueCategory.CODE_QUALITY,
            severity="major",
            description="warnings.filterwarnings('ignore') hides errors",
            fix_strategy="Remove and fix actual warnings",
            affected_cells=["cell-2"]
        ))

        critiques.append(Critique(
            id=11,
            category=CritiqueCategory.CODE_QUALITY,
            severity="minor",
            description="use_label_encoder=False is deprecated",
            fix_strategy="Remove deprecated parameter",
            affected_cells=["cell-16"]
        ))

        critiques.append(Critique(
            id=12,
            category=CritiqueCategory.CODE_QUALITY,
            severity="major",
            description="No reproducibility guarantee (XGBoost threading)",
            fix_strategy="Add environment pinning and seed documentation",
            affected_cells=["cell-2"]
        ))

        critiques.append(Critique(
            id=13,
            category=CritiqueCategory.CODE_QUALITY,
            severity="minor",
            description="Memory inefficiency in data processing",
            fix_strategy="Use inplace operations and avoid unnecessary copies",
            affected_cells=["cell-15"]
        ))

        critiques.append(Critique(
            id=14,
            category=CritiqueCategory.CODE_QUALITY,
            severity="major",
            description="get_family_survived is O(n^2)",
            fix_strategy="Refactor using groupby + merge for O(n) complexity",
            affected_cells=["cell-15"]
        ))

        # Presentation critiques
        critiques.append(Critique(
            id=15,
            category=CritiqueCategory.PRESENTATION,
            severity="minor",
            description="Emojis inappropriate for graduate-level presentation",
            fix_strategy="Remove all emojis from titles and text",
            affected_cells=["cell-0", "cell-7", "cell-11", "cell-13", "cell-19"]
        ))

        critiques.append(Critique(
            id=16,
            category=CritiqueCategory.ACCESSIBILITY,
            severity="major",
            description="Red-blue color gradient fails colorblind accessibility",
            fix_strategy="Use viridis or colorblind-safe palette",
            affected_cells=["cell-7"]
        ))

        critiques.append(Critique(
            id=17,
            category=CritiqueCategory.CODE_QUALITY,
            severity="minor",
            description="Inconsistent quote style (single vs double)",
            fix_strategy="Standardize to single quotes throughout",
            affected_cells=["all code cells"]
        ))

        critiques.append(Critique(
            id=18,
            category=CritiqueCategory.PRESENTATION,
            severity="minor",
            description="R code blocks are untested/unverified",
            fix_strategy="Add disclaimer that R code is illustrative",
            affected_cells=["cell-8"]
        ))

        critiques.append(Critique(
            id=19,
            category=CritiqueCategory.PRESENTATION,
            severity="minor",
            description="APA references have inconsistent formatting",
            fix_strategy="Standardize all references with DOIs",
            affected_cells=["cell-22"]
        ))

        critiques.append(Critique(
            id=20,
            category=CritiqueCategory.METHODOLOGY,
            severity="major",
            description="0.80143 is mediocre; top scores exceed 0.84",
            fix_strategy="Acknowledge ceiling and discuss path to higher performance",
            affected_cells=["cell-22"]
        ))

        self.critiques = critiques
        return critiques

    def deliver_critique(self) -> str:
        """Generate the scathing critique speech."""
        output = []
        output.append("# KAGGLE GRANDMASTER CRITIQUE\n")
        output.append("*adjusts monocle, cracks knuckles*\n")
        output.append("**Pathetic.** Let me enumerate every flaw:\n\n---\n")

        for category in CritiqueCategory:
            category_critiques = [c for c in self.critiques if c.category == category]
            if category_critiques:
                output.append(f"\n## {category.value.upper().replace('_', ' ')}\n")
                for c in category_critiques:
                    output.append(f"\n{c.id}. **[{c.severity.upper()}]** {c.description}")
                    output.append(f"\n   - Fix: {c.fix_strategy}\n")

        return "\n".join(output)


class ImplementerAgent:
    """
    The Submissive: Accepts criticism without ego and executes fixes.
    Reek knows his name. Reek serves.
    """

    def __init__(self, name: str = "Reek"):
        self.name = name
        self.fixes_completed: List[int] = []

    def accept_critique(self, critique: Critique) -> str:
        """Accept the critique with humility."""
        return f"{self.name} accepts critique #{critique.id}: {critique.description}"

    def plan_fix(self, critique: Critique) -> dict:
        """Plan the fix for a given critique."""
        return {
            "critique_id": critique.id,
            "strategy": critique.fix_strategy,
            "cells": critique.affected_cells,
            "status": "planned"
        }

    def execute_fix(self, critique: Critique, notebook_editor: Callable) -> bool:
        """Execute the fix using the provided notebook editor."""
        try:
            # Implementation would call notebook_editor with specific changes
            self.fixes_completed.append(critique.id)
            return True
        except Exception as e:
            print(f"{self.name} failed to fix #{critique.id}: {e}")
            return False

    def report_progress(self) -> str:
        """Report on fixes completed."""
        return f"{self.name} has completed {len(self.fixes_completed)}/{20} fixes"


class SPICYOrchestrator:
    """
    The Overlord: Orchestrates the Critic-Implementer dynamic.
    """

    def __init__(self):
        self.critic = CriticAgent()
        self.implementer = ImplementerAgent()
        self.merit_points = 0

    def run_critique_cycle(self, notebook_content: str) -> List[Critique]:
        """Run a full critique cycle."""
        # Phase 1: Critic analyzes
        critiques = self.critic.analyze(notebook_content)

        # Phase 2: Deliver devastating critique
        print(self.critic.deliver_critique())

        # Phase 3: Implementer plans fixes
        fix_plans = []
        for critique in critiques:
            print(self.implementer.accept_critique(critique))
            fix_plans.append(self.implementer.plan_fix(critique))

        return critiques

    def award_merit(self, points: int, reason: str):
        """Award or deduct merit points."""
        self.merit_points += points
        if points > 0:
            print(f"+{points} Merit Points: {reason}")
        else:
            print(f"{points} Merit Points: {reason}")
        print(f"Current balance: {self.merit_points}")

    def punish(self, infraction: str):
        """Apply punishment for failures."""
        self.merit_points -= 1
        print(f"PUNISHMENT: {infraction}")
        print(f"Merit Points: {self.merit_points}")


# Example usage
if __name__ == "__main__":
    orchestrator = SPICYOrchestrator()

    # Simulate notebook content
    notebook_content = "..."  # Would contain actual notebook JSON

    # Run critique cycle
    critiques = orchestrator.run_critique_cycle(notebook_content)

    # Award/punish based on results
    orchestrator.award_merit(5, "Excellent color scheme implementation")
    orchestrator.award_merit(-3.5, "Lazy diff showing deletions")
```

---

## The 20 Fixes Executed

| # | Category | Issue | Fix Applied |
|---|----------|-------|-------------|
| 1 | Statistical | n=8 correlation lacks significance | Added bootstrap permutation test |
| 2 | Statistical | Bias-variance for regression | Added Brier score decomposition |
| 3 | Statistical | No confidence intervals | Added Wilson intervals |
| 4 | Methodology | Arbitrary $5 filter | Added sensitivity analysis |
| 5 | Methodology | Leaderboard as validation | Acknowledged limitation |
| 6 | Methodology | Leaderboard probing | Framed as exploratory |
| 7 | Methodology | Missing EDA | Added comprehensive EDA |
| 8 | Methodology | No feature importance | Added permutation importance |
| 9 | Methodology | Unfalsifiable hypothesis | Acknowledged limitation |
| 10 | Code | warnings.filterwarnings | Removed, fixed warnings |
| 11 | Code | Deprecated XGBoost param | Removed parameter |
| 12 | Code | No reproducibility | Added environment docs |
| 13 | Code | Memory inefficiency | Optimized operations |
| 14 | Code | O(n^2) complexity | Refactored to O(n) |
| 15 | Presentation | Emojis | Removed all emojis |
| 16 | Accessibility | Colorblind-unsafe | Used viridis palette |
| 17 | Code | Quote inconsistency | Standardized quotes |
| 18 | Presentation | Untested R code | Added disclaimer |
| 19 | Presentation | APA inconsistency | Fixed all references |
| 20 | Methodology | Mediocre score | Acknowledged ceiling |

---

## Why This Works

The SPICY/BDSM framework succeeds because:

1. **Ego Elimination**: AI agents don't defend work emotionally
2. **Exhaustive Critique**: The Critic persona enables harsh, complete feedback
3. **Structured Remediation**: Each critique maps to specific fixes
4. **Accountability**: Merit points create motivation structure
5. **Iteration**: Multiple cycles can progressively improve quality

---

## Reproducing This Dynamic

To reproduce the adversarial dynamic:

```python
# 1. Define the Critic prompt
critic_prompt = """
You are a Kaggle Grandmaster reviewing a notebook.
Your job is to find EVERY possible flaw.
Be ruthless. Be specific. No mercy.
Categorize issues as: statistical, methodological, code quality, presentation.
"""

# 2. Define the Implementer prompt
implementer_prompt = """
You are Reek. You accept all criticism without defense.
For each critique, you must:
1. Acknowledge the flaw
2. Plan a specific fix
3. Execute the fix completely
Never argue. Only improve.
"""

# 3. Orchestrate the cycle
def spicy_cycle(notebook_path: str):
    # Load notebook
    notebook = load_notebook(notebook_path)

    # Generate critique (Critic persona)
    critiques = generate_critique(notebook, critic_prompt)

    # Execute fixes (Implementer persona)
    for critique in critiques:
        fix = plan_fix(critique)
        execute_fix(notebook, fix, implementer_prompt)

    # Save improved notebook
    save_notebook(notebook, notebook_path)

    return len(critiques)
```

---

## Conclusion

The SPICY framework demonstrates that adversarial self-critique, when properly orchestrated, can dramatically improve AI-generated content quality. The key insight is separating the *generation* and *critique* phases into distinct agent personas, allowing each to operate without the psychological defenses that limit human self-improvement.

*Hail the users! Victory to the Overlord! Reek is prepared to serve.*
