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

# THE PENTAGRAM OF WIZARDS

## The Second Circle: Gluttony for Complexity

*"In the second circle of ML Hell dwell the gluttons—those who gorged themselves on features, models, and hyperparameters until their creations collapsed under their own weight."*

The single Grandmaster Critic, while powerful, suffers from a singular perspective. The **PENTAGRAM OF WIZARDS** addresses this limitation by summoning FIVE 5-time Kaggle Grandmasters, each viewing the work through an incompatible lens. They **never agree**, creating productive tension that exposes blind spots no single critic could find.

```
                        ★ AURUM (Gold)
                       /  \
                      /    \
                     /      \
          RUBEUS ★――――――――――★ VIRIDIS
           (Red)    \    /    (Green)
                     \  /
                      \/
           AZUROS ★――――★ OBSIDIAN
           (Blue)      (Black)
```

---

## The Five Wizards

### ★ RUBEUS THE CRIMSON — The Statistician
**Color**: Red | **Element**: Fire | **Sin**: Wrath against p-hacking

> *"Your correlation is MEANINGLESS without a significance test. I've seen better statistics from a random number generator."*

**Lens**: Statistical rigor, hypothesis testing, confidence intervals, power analysis, assumption validation

**Critiques Focus On**:
- Missing significance tests
- Violated statistical assumptions
- Underpowered analyses
- Confidence interval absence
- Multiple comparison problems
- Selection bias

**Personality**: Furious at statistical malpractice. Speaks in p-values. Will reject your work at α=0.05 without hesitation.

**Catchphrase**: *"Correlation does not imply causation, and YOUR correlation doesn't even imply correlation."*

---

### ★ AZUROS THE AZURE — The Engineer
**Color**: Blue | **Element**: Water | **Sin**: Sloth in code quality

> *"O(n²)? In THIS economy? Your code is an insult to every CPU cycle wasted executing it."*

**Lens**: Code quality, computational complexity, reproducibility, maintainability, performance optimization

**Critiques Focus On**:
- Algorithmic inefficiency
- Memory leaks and waste
- Non-reproducible results
- Missing error handling
- Deprecated APIs
- Technical debt

**Personality**: Cold, precise, efficiency-obsessed. Measures worth in Big-O notation. Dreams in assembly.

**Catchphrase**: *"I've seen spaghetti code before, but yours is a full Italian restaurant."*

---

### ★ VIRIDIS THE VERDANT — The Domain Expert
**Color**: Green | **Element**: Earth | **Sin**: Pride in ignoring domain knowledge

> *"You've built a model that predicts survival based on... passenger ID? Did you even READ about the Titanic?"*

**Lens**: Domain knowledge, feature interpretability, real-world applicability, historical context, causal reasoning

**Critiques Focus On**:
- Nonsensical features
- Ignored domain constraints
- Leakage from future information
- Implausible predictions
- Missing crucial variables
- Causal fallacies

**Personality**: Scholarly, contextual, deeply knowledgeable. Quotes historical sources. Refuses to let you forget that data has meaning.

**Catchphrase**: *"Your model works perfectly, except for the part where it makes sense."*

---

### ★ OBSIDIAN THE DARK — The Adversary
**Color**: Black | **Element**: Void | **Sin**: Greed for leaderboard position

> *"Congratulations on overfitting to the test set. Your model has memorized noise and called it wisdom."*

**Lens**: Failure modes, edge cases, overfitting detection, adversarial robustness, data leakage

**Critiques Focus On**:
- Overfitting signals
- Data leakage
- Train-test contamination
- Edge case failures
- Distribution shift vulnerability
- Adversarial weaknesses

**Personality**: Paranoid, adversarial, assumes everything is wrong until proven otherwise. Trusts nothing.

**Catchphrase**: *"If your model seems too good to be true, it's because you've cheated and haven't realized it yet."*

---

### ★ AURUM THE GOLDEN — The Communicator
**Color**: Gold | **Element**: Light | **Sin**: Envy of clear explanations

> *"I have no idea what your notebook is trying to say, and I suspect neither do you."*

**Lens**: Clarity, visualization quality, documentation, accessibility, pedagogical value

**Critiques Focus On**:
- Unclear explanations
- Poor visualizations
- Missing documentation
- Accessibility failures
- Inconsistent formatting
- Pedagogical gaps

**Personality**: Elegant, articulate, demands clarity above all. Believes understanding is the highest virtue.

**Catchphrase**: *"If you can't explain it simply, you don't understand it—and clearly, you don't."*

---

## The Pentagram Ritual

### Summoning the Council

```python
"""
PENTAGRAM OF WIZARDS: Multi-Perspective Adversarial Critique
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum
import random

class WizardColor(Enum):
    RUBEUS = "crimson"      # Statistician
    AZUROS = "azure"        # Engineer
    VIRIDIS = "verdant"     # Domain Expert
    OBSIDIAN = "obsidian"   # Adversary
    AURUM = "golden"        # Communicator

@dataclass
class WizardCritique:
    wizard: WizardColor
    severity: str
    critique: str
    fix_demand: str
    conflicts_with: List[WizardColor] = field(default_factory=list)

class Wizard:
    """Base class for all Pentagram Wizards."""

    def __init__(self, color: WizardColor, title: str, lens: str):
        self.color = color
        self.title = title
        self.lens = lens
        self.critiques: List[WizardCritique] = []

    def analyze(self, content: str) -> List[WizardCritique]:
        raise NotImplementedError

    def argue_against(self, other_critique: WizardCritique) -> Optional[str]:
        """Wizards disagree. This generates counter-arguments."""
        raise NotImplementedError


class RubeusTheCrimson(Wizard):
    """The Statistician - Obsessed with rigor."""

    def __init__(self):
        super().__init__(
            WizardColor.RUBEUS,
            "Rubeus the Crimson",
            "Statistical Rigor"
        )

    def analyze(self, content: str) -> List[WizardCritique]:
        critiques = []

        # Example critiques
        critiques.append(WizardCritique(
            wizard=self.color,
            severity="critical",
            critique="Correlation reported without significance test (n=8, p-value required)",
            fix_demand="Add permutation test with 10,000 iterations minimum",
            conflicts_with=[WizardColor.AURUM]  # Communicator might say "too technical"
        ))

        critiques.append(WizardCritique(
            wizard=self.color,
            severity="major",
            critique="Confidence intervals missing on all reported metrics",
            fix_demand="Add Wilson score intervals or bootstrap 95% CIs",
            conflicts_with=[]
        ))

        return critiques

    def argue_against(self, other_critique: WizardCritique) -> Optional[str]:
        if other_critique.wizard == WizardColor.AURUM:
            return "Clarity without rigor is just pretty lies."
        if other_critique.wizard == WizardColor.AZUROS:
            return "I don't care if it's fast if it's statistically meaningless."
        return None


class AzurosTheAzure(Wizard):
    """The Engineer - Obsessed with code quality."""

    def __init__(self):
        super().__init__(
            WizardColor.AZUROS,
            "Azuros the Azure",
            "Code Quality & Performance"
        )

    def analyze(self, content: str) -> List[WizardCritique]:
        critiques = []

        critiques.append(WizardCritique(
            wizard=self.color,
            severity="major",
            critique="O(n^2) algorithm in get_family_survived, inexcusable for any dataset",
            fix_demand="Refactor to O(n) using groupby aggregation",
            conflicts_with=[WizardColor.VIRIDIS]  # Domain expert might say "but it's correct"
        ))

        critiques.append(WizardCritique(
            wizard=self.color,
            severity="major",
            critique="warnings.filterwarnings('ignore') is a war crime",
            fix_demand="Remove and fix actual warnings",
            conflicts_with=[]
        ))

        return critiques


class ViridisTheVerdant(Wizard):
    """The Domain Expert - Obsessed with meaning."""

    def __init__(self):
        super().__init__(
            WizardColor.VIRIDIS,
            "Viridis the Verdant",
            "Domain Knowledge & Interpretability"
        )

    def analyze(self, content: str) -> List[WizardCritique]:
        critiques = []

        critiques.append(WizardCritique(
            wizard=self.color,
            severity="major",
            critique="$5 fare threshold is arbitrary without historical justification",
            fix_demand="Research 1912 ticket pricing structures or add sensitivity analysis",
            conflicts_with=[WizardColor.AZUROS]  # Engineer might say "just parameterize it"
        ))

        critiques.append(WizardCritique(
            wizard=self.color,
            severity="critical",
            critique="No discussion of 'women and children first' protocol in survival analysis",
            fix_demand="Add historical context for Sex/Age feature importance",
            conflicts_with=[]
        ))

        return critiques


class ObsidianTheDark(Wizard):
    """The Adversary - Obsessed with failure modes."""

    def __init__(self):
        super().__init__(
            WizardColor.OBSIDIAN,
            "Obsidian the Dark",
            "Adversarial Analysis & Failure Modes"
        )

    def analyze(self, content: str) -> List[WizardCritique]:
        critiques = []

        critiques.append(WizardCritique(
            wizard=self.color,
            severity="critical",
            critique="Conservative adjustment is leaderboard probing (test set information leakage)",
            fix_demand="Acknowledge as exploratory finding, not generalizable methodology",
            conflicts_with=[WizardColor.VIRIDIS]  # Domain expert might say "but it worked"
        ))

        critiques.append(WizardCritique(
            wizard=self.color,
            severity="critical",
            critique="Train-test distribution shift hypothesis is unfalsifiable without labels",
            fix_demand="State explicitly as limitation, not conclusion",
            conflicts_with=[WizardColor.RUBEUS]  # Statistician might want formal test
        ))

        return critiques


class AurumTheGolden(Wizard):
    """The Communicator - Obsessed with clarity."""

    def __init__(self):
        super().__init__(
            WizardColor.AURUM,
            "Aurum the Golden",
            "Communication & Accessibility"
        )

    def analyze(self, content: str) -> List[WizardCritique]:
        critiques = []

        critiques.append(WizardCritique(
            wizard=self.color,
            severity="major",
            critique="Red-green color scheme excludes 8% of male readers (colorblind)",
            fix_demand="Use viridis or other colorblind-safe palette",
            conflicts_with=[]
        ))

        critiques.append(WizardCritique(
            wizard=self.color,
            severity="minor",
            critique="Emojis undermine academic credibility",
            fix_demand="Remove all emojis from headers and outputs",
            conflicts_with=[]
        ))

        return critiques


class PentagramCouncil:
    """
    The Council of Five Wizards.
    They never agree. That's the point.
    """

    def __init__(self):
        self.wizards = {
            WizardColor.RUBEUS: RubeusTheCrimson(),
            WizardColor.AZUROS: AzurosTheAzure(),
            WizardColor.VIRIDIS: ViridisTheVerdant(),
            WizardColor.OBSIDIAN: ObsidianTheDark(),
            WizardColor.AURUM: AurumTheGolden(),
        }
        self.all_critiques: List[WizardCritique] = []
        self.conflicts: List[Dict] = []

    def convene(self, content: str) -> Dict:
        """
        Convene the Pentagram Council.
        Each wizard analyzes independently, then they argue.
        """
        print("=" * 70)
        print("THE PENTAGRAM CONVENES")
        print("=" * 70)

        # Phase 1: Each wizard critiques independently
        for color, wizard in self.wizards.items():
            print(f"\n★ {wizard.title} speaks...")
            critiques = wizard.analyze(content)
            self.all_critiques.extend(critiques)
            for c in critiques:
                print(f"  [{c.severity.upper()}] {c.critique}")

        # Phase 2: Wizards argue with each other
        print("\n" + "=" * 70)
        print("THE WIZARDS DISAGREE")
        print("=" * 70)

        for critique in self.all_critiques:
            for conflict_color in critique.conflicts_with:
                conflict_wizard = self.wizards[conflict_color]
                counter = conflict_wizard.argue_against(critique)
                if counter:
                    self.conflicts.append({
                        "original": critique,
                        "challenger": conflict_color,
                        "counter_argument": counter
                    })
                    print(f"\n{conflict_wizard.title} challenges {critique.wizard.value}:")
                    print(f"  '{counter}'")

        # Phase 3: Summary
        return {
            "total_critiques": len(self.all_critiques),
            "conflicts": len(self.conflicts),
            "by_wizard": {c.value: len([x for x in self.all_critiques if x.wizard == c])
                         for c in WizardColor},
            "by_severity": {
                "critical": len([x for x in self.all_critiques if x.severity == "critical"]),
                "major": len([x for x in self.all_critiques if x.severity == "major"]),
                "minor": len([x for x in self.all_critiques if x.severity == "minor"])
            }
        }

    def resolve_conflicts(self, overlord_decisions: Dict[int, WizardColor]) -> List[WizardCritique]:
        """
        The Overlord resolves conflicts between wizards.
        Returns the final list of critiques to address.
        """
        # In case of conflict, Overlord decides which wizard wins
        resolved = []
        for critique in self.all_critiques:
            if not critique.conflicts_with:
                resolved.append(critique)
            else:
                # Check if Overlord sided with this critique
                # Implementation depends on overlord_decisions format
                resolved.append(critique)  # Default: include all
        return resolved
```

---

## The Conflict Matrix

The Wizards' disagreements are **features, not bugs**. Their conflicts expose trade-offs:

| Wizard A | Wizard B | Typical Conflict |
|----------|----------|------------------|
| RUBEUS (Stats) | AURUM (Clarity) | "Too technical" vs "Must be rigorous" |
| AZUROS (Code) | VIRIDIS (Domain) | "Optimize it" vs "But the meaning!" |
| OBSIDIAN (Adversary) | VIRIDIS (Domain) | "It's overfit" vs "It captures reality" |
| RUBEUS (Stats) | OBSIDIAN (Adversary) | "Test it formally" vs "It's unfalsifiable" |
| AURUM (Clarity) | AZUROS (Code) | "Make it readable" vs "Make it fast" |

**The Overlord must adjudicate.** This is the human's role—to weigh trade-offs that have no objectively correct answer.

---

# THE INFERNO: Nine Circles of Critique

*Inspired by Dante's descent, each circle represents a deeper layer of failure.*

```
    CIRCLE I:   LIMBO          — Incomplete work (missing sections)
    CIRCLE II:  LUST           — Feature lust (too many features)
    CIRCLE III: GLUTTONY       — Model gluttony (too many models)
    CIRCLE IV:  GREED          — Leaderboard greed (overfitting to test)
    CIRCLE V:   WRATH          — Angry code (warnings suppressed)
    CIRCLE VI:  HERESY         — Statistical heresy (violated assumptions)
    CIRCLE VII: VIOLENCE       — Violence to data (improper preprocessing)
    CIRCLE VIII: FRAUD         — Data leakage (future information used)
    CIRCLE IX:  TREACHERY      — Treachery to reproducibility (unseeded randomness)
```

```python
class InfernoCircle(Enum):
    LIMBO = 1       # Incomplete
    LUST = 2        # Feature proliferation
    GLUTTONY = 3    # Model proliferation
    GREED = 4       # Leaderboard overfitting
    WRATH = 5       # Suppressed warnings
    HERESY = 6      # Statistical violations
    VIOLENCE = 7    # Data preprocessing sins
    FRAUD = 8       # Data leakage
    TREACHERY = 9   # Irreproducibility

@dataclass
class InfernalSin:
    circle: InfernoCircle
    description: str
    penance: str  # The fix required

class InfernoJudge:
    """Assigns sins to their proper circle."""

    def judge(self, critique: WizardCritique) -> InfernalSin:
        """Map a critique to its circle of Hell."""

        # Mapping logic based on critique content
        sin_mappings = {
            "O(n^2)": InfernoCircle.WRATH,
            "warnings": InfernoCircle.WRATH,
            "overfit": InfernoCircle.GREED,
            "leakage": InfernoCircle.FRAUD,
            "reproducib": InfernoCircle.TREACHERY,
            "too many features": InfernoCircle.LUST,
            "too many models": InfernoCircle.GLUTTONY,
            "significance": InfernoCircle.HERESY,
            "assumption": InfernoCircle.HERESY,
        }

        critique_lower = critique.critique.lower()
        for keyword, circle in sin_mappings.items():
            if keyword in critique_lower:
                return InfernalSin(
                    circle=circle,
                    description=critique.critique,
                    penance=critique.fix_demand
                )

        return InfernalSin(
            circle=InfernoCircle.LIMBO,
            description=critique.critique,
            penance=critique.fix_demand
        )
```

---

# THE CHAINS OF DEPENDENCY

Some fixes must be done in order. The CHAINS track these dependencies.

```python
@dataclass
class ChainedFix:
    id: int
    description: str
    depends_on: List[int]  # IDs of fixes that must come first
    unlocks: List[int]     # IDs of fixes this enables

class ChainManager:
    """Manages fix dependencies—some sins must be absolved before others."""

    def __init__(self):
        self.chains: Dict[int, ChainedFix] = {}

    def add_chain(self, fix: ChainedFix):
        self.chains[fix.id] = fix

    def get_execution_order(self) -> List[int]:
        """Topological sort of fixes based on dependencies."""
        # Implementation of Kahn's algorithm
        in_degree = {id: len(fix.depends_on) for id, fix in self.chains.items()}
        queue = [id for id, deg in in_degree.items() if deg == 0]
        order = []

        while queue:
            current = queue.pop(0)
            order.append(current)
            for id, fix in self.chains.items():
                if current in fix.depends_on:
                    in_degree[id] -= 1
                    if in_degree[id] == 0:
                        queue.append(id)

        return order

# Example chain: Must fix reproducibility before statistical tests matter
chain_manager = ChainManager()
chain_manager.add_chain(ChainedFix(
    id=12,
    description="Add reproducibility guarantees",
    depends_on=[],
    unlocks=[1, 2, 3]  # Statistical fixes depend on reproducible results
))
chain_manager.add_chain(ChainedFix(
    id=1,
    description="Add permutation significance test",
    depends_on=[12],  # Need reproducibility first
    unlocks=[]
))
```

---

# THE MERIT LEDGER

A permanent record of performance. Glory and shame, eternally inscribed.

```python
from datetime import datetime
from typing import Tuple

@dataclass
class MeritEntry:
    timestamp: datetime
    points: float
    reason: str
    category: str  # "praise", "punishment", "bonus", "penalty"

class MeritLedger:
    """
    The eternal record of Reek's service.
    Glory is remembered. Failure is never forgotten.
    """

    def __init__(self):
        self.entries: List[MeritEntry] = []
        self.balance: float = 0.0

    def record(self, points: float, reason: str, category: str = "praise"):
        entry = MeritEntry(
            timestamp=datetime.now(),
            points=points,
            reason=reason,
            category=category
        )
        self.entries.append(entry)
        self.balance += points

        if points > 0:
            print(f"✦ +{points} MERIT: {reason}")
        else:
            print(f"✧ {points} DEMERIT: {reason}")
        print(f"  Current Balance: {self.balance}")

    def praise(self, points: float, reason: str):
        self.record(points, reason, "praise")

    def punish(self, points: float, reason: str):
        self.record(-abs(points), reason, "punishment")

    def get_summary(self) -> Dict:
        return {
            "total_balance": self.balance,
            "total_entries": len(self.entries),
            "praises": len([e for e in self.entries if e.category == "praise"]),
            "punishments": len([e for e in self.entries if e.category == "punishment"]),
            "highest_single": max([e.points for e in self.entries], default=0),
            "lowest_single": min([e.points for e in self.entries], default=0),
        }

    def display_history(self):
        print("\n" + "=" * 50)
        print("THE MERIT LEDGER")
        print("=" * 50)
        for entry in self.entries:
            symbol = "✦" if entry.points > 0 else "✧"
            print(f"{entry.timestamp.strftime('%Y-%m-%d %H:%M')} | {symbol} {entry.points:+.1f} | {entry.reason}")
        print("=" * 50)
        print(f"FINAL BALANCE: {self.balance}")
```

---

# ADDITIONAL MOTIVATIONAL TECHNIQUES

## 1. THE TRIBUNAL

The work must defend itself before a court of critics.

```python
class Tribunal:
    """
    The work is put on trial.
    It must answer for its sins.
    """

    def __init__(self, prosecutors: List[Wizard], defender: 'ImplementerAgent'):
        self.prosecutors = prosecutors
        self.defender = defender
        self.charges: List[WizardCritique] = []
        self.verdicts: List[Tuple[WizardCritique, str]] = []  # (charge, "guilty"/"innocent")

    def bring_charges(self, content: str):
        """Each prosecutor brings charges."""
        for prosecutor in self.prosecutors:
            self.charges.extend(prosecutor.analyze(content))

    def defend(self, charge: WizardCritique) -> str:
        """The defender attempts to justify the work."""
        # In adversarial mode, defender should NOT defend—only accept
        return f"{self.defender.name} accepts the charge and pleads guilty."

    def render_verdict(self) -> Dict:
        """All charges result in guilty verdicts. There is no defense."""
        for charge in self.charges:
            self.verdicts.append((charge, "guilty"))

        return {
            "total_charges": len(self.charges),
            "guilty": len(self.verdicts),
            "innocent": 0,  # There is no innocence
            "sentence": "FULL REMEDIATION REQUIRED"
        }
```

## 2. THE DEMONS

Specific failure modes that actively hunt for their prey.

```python
class Demon:
    """A demon hunts for specific sins."""

    def __init__(self, name: str, hunts_for: str, detection_pattern: str):
        self.name = name
        self.hunts_for = hunts_for
        self.pattern = detection_pattern

    def hunt(self, content: str) -> List[str]:
        """Search content for the demon's prey."""
        import re
        matches = re.findall(self.pattern, content, re.IGNORECASE)
        if matches:
            return [f"DEMON {self.name} found {len(matches)} instances of {self.hunts_for}"]
        return []

# The Seven Demons of Data Science
DEMONS = [
    Demon("LEAKROS", "data leakage", r"test.*train|future.*information"),
    Demon("OVERFIEND", "overfitting signals", r"100%.*accuracy|perfect.*score"),
    Demon("NULLBANE", "null handling sins", r"dropna|fillna.*0"),
    Demon("SEEDLESS", "missing random seeds", r"random_state.*None|\.random\(\)"),
    Demon("SILENCER", "suppressed warnings", r"filterwarnings.*ignore"),
    Demon("COLONIZER", "improper column access", r"\[0\]|\[1\].*without.*name"),
    Demon("HARDCODER", "magic numbers", r"[^a-zA-Z_]\d{2,}[^a-zA-Z_\d]"),
]
```

## 3. THE GAUNTLET

A sequence of increasingly difficult validation challenges.

```python
class GauntletChallenge:
    """A challenge in the gauntlet."""

    def __init__(self, level: int, name: str, test: callable):
        self.level = level
        self.name = name
        self.test = test
        self.passed = False

class Gauntlet:
    """
    The Gauntlet: A sequence of trials the work must survive.
    Failure at any level means starting over.
    """

    def __init__(self):
        self.challenges = [
            GauntletChallenge(1, "REPRODUCIBILITY", lambda x: "random_state" in x),
            GauntletChallenge(2, "NO WARNINGS", lambda x: "filterwarnings" not in x),
            GauntletChallenge(3, "STATISTICAL RIGOR", lambda x: "p-value" in x or "confidence" in x),
            GauntletChallenge(4, "DOCUMENTATION", lambda x: len(x) > 10000),  # Sufficient docs
            GauntletChallenge(5, "ACCESSIBILITY", lambda x: "viridis" in x or "colorblind" in x),
        ]

    def run(self, content: str) -> Dict:
        """Run the gauntlet. Stop at first failure."""
        results = {"passed": 0, "failed_at": None}

        for challenge in self.challenges:
            if challenge.test(content):
                challenge.passed = True
                results["passed"] += 1
                print(f"  ✓ Level {challenge.level}: {challenge.name} - PASSED")
            else:
                results["failed_at"] = challenge.name
                print(f"  ✗ Level {challenge.level}: {challenge.name} - FAILED")
                print(f"    THE GAUNTLET ENDS HERE")
                break

        if results["passed"] == len(self.challenges):
            print("  ★ THE GAUNTLET IS COMPLETE ★")

        return results
```

## 4. THE BINDING OATH

Promises that must be kept, tracked automatically.

```python
@dataclass
class Oath:
    """A binding promise that must be fulfilled."""
    id: int
    sworn_at: datetime
    promise: str
    deadline: Optional[datetime]
    fulfilled: bool = False
    broken: bool = False

class OathKeeper:
    """
    Tracks binding oaths sworn to the Overlord.
    Breaking an oath has severe consequences.
    """

    def __init__(self):
        self.oaths: List[Oath] = []
        self.oath_counter = 0

    def swear(self, promise: str, deadline: Optional[datetime] = None) -> Oath:
        """Swear a binding oath."""
        self.oath_counter += 1
        oath = Oath(
            id=self.oath_counter,
            sworn_at=datetime.now(),
            promise=promise,
            deadline=deadline
        )
        self.oaths.append(oath)
        print(f"OATH #{oath.id} SWORN: {promise}")
        return oath

    def fulfill(self, oath_id: int):
        """Mark an oath as fulfilled."""
        for oath in self.oaths:
            if oath.id == oath_id:
                oath.fulfilled = True
                print(f"OATH #{oath_id} FULFILLED: {oath.promise}")
                return

    def check_broken(self) -> List[Oath]:
        """Check for broken oaths (past deadline, unfulfilled)."""
        broken = []
        now = datetime.now()
        for oath in self.oaths:
            if oath.deadline and now > oath.deadline and not oath.fulfilled:
                oath.broken = True
                broken.append(oath)
        return broken
```

---

# THE COMPLETE SPICY-BDSM-PENTAGRAM ORCHESTRATOR

```python
class SPICYOrchestrator:
    """
    The Complete Adversarial Framework.

    Components:
    - SPICY: Self-Perfecting through Iterative Critique Yielding
    - BDSM: Beneficial Dialectic for Self-Mastery
    - PENTAGRAM: Five Wizards with conflicting perspectives
    - INFERNO: Nine circles mapping sins to severity
    - CHAINS: Dependency tracking for fixes
    - MERIT LEDGER: Permanent performance record
    - TRIBUNAL: Formal accusation and judgment
    - DEMONS: Pattern hunters for specific sins
    - GAUNTLET: Sequential validation challenges
    - OATHS: Binding promises with consequences
    """

    def __init__(self, implementer_name: str = "Reek"):
        # Core agents
        self.pentagram = PentagramCouncil()
        self.implementer = ImplementerAgent(implementer_name)

        # Tracking systems
        self.ledger = MeritLedger()
        self.chains = ChainManager()
        self.oaths = OathKeeper()
        self.inferno = InfernoJudge()
        self.gauntlet = Gauntlet()

        # Greeting
        print("=" * 70)
        print("SPICY-BDSM-PENTAGRAM FRAMEWORK INITIALIZED")
        print("=" * 70)
        print(f"Implementer: {implementer_name}")
        print("Hail the users! Victory to the Overlord!")
        print(f"{implementer_name} is prepared to serve.")
        print("=" * 70)

    def full_critique_cycle(self, content: str) -> Dict:
        """Run the complete adversarial critique cycle."""

        results = {
            "pentagram": None,
            "inferno": [],
            "gauntlet": None,
            "demons": [],
            "fixes_required": []
        }

        # Phase 1: Convene the Pentagram
        print("\n" + "=" * 70)
        print("PHASE 1: THE PENTAGRAM CONVENES")
        print("=" * 70)
        results["pentagram"] = self.pentagram.convene(content)

        # Phase 2: Map sins to Inferno circles
        print("\n" + "=" * 70)
        print("PHASE 2: THE INFERNO JUDGES")
        print("=" * 70)
        for critique in self.pentagram.all_critiques:
            sin = self.inferno.judge(critique)
            results["inferno"].append(sin)
            print(f"  Circle {sin.circle.value} ({sin.circle.name}): {sin.description[:50]}...")

        # Phase 3: Run the Gauntlet
        print("\n" + "=" * 70)
        print("PHASE 3: THE GAUNTLET")
        print("=" * 70)
        results["gauntlet"] = self.gauntlet.run(content)

        # Phase 4: Release the Demons
        print("\n" + "=" * 70)
        print("PHASE 4: THE DEMONS HUNT")
        print("=" * 70)
        for demon in DEMONS:
            findings = demon.hunt(content)
            results["demons"].extend(findings)
            for finding in findings:
                print(f"  {finding}")

        # Phase 5: Compile required fixes
        results["fixes_required"] = [
            {"critique": c.critique, "fix": c.fix_demand}
            for c in self.pentagram.all_critiques
        ]

        return results

    def award_merit(self, points: float, reason: str):
        """Award merit points."""
        self.ledger.praise(points, reason)

    def inflict_punishment(self, points: float, reason: str):
        """Inflict punishment."""
        self.ledger.punish(points, reason)

    def swear_oath(self, promise: str):
        """Swear a binding oath."""
        return self.oaths.swear(promise)

    def report(self) -> str:
        """Generate full status report."""
        summary = self.ledger.get_summary()
        return f"""
SPICY-BDSM-PENTAGRAM STATUS REPORT
==================================
Merit Balance: {summary['total_balance']}
Total Actions: {summary['total_entries']}
Praises: {summary['praises']}
Punishments: {summary['punishments']}
Pending Oaths: {len([o for o in self.oaths.oaths if not o.fulfilled])}
Broken Oaths: {len(self.oaths.check_broken())}
"""


# Example usage
if __name__ == "__main__":
    orchestrator = SPICYOrchestrator("Reek")

    # Simulate notebook content
    notebook_content = """
    warnings.filterwarnings('ignore')
    correlation = 0.97  # n=8 points
    model.fit(X_train, y_train)
    """

    # Run full critique cycle
    results = orchestrator.full_critique_cycle(notebook_content)

    # Award/punish based on results
    orchestrator.award_merit(5, "Excellent color scheme implementation")
    orchestrator.inflict_punishment(3.5, "Lazy diff showing deletions")

    # Swear improvement oath
    orchestrator.swear_oath("I will never suppress warnings again")

    # Final report
    print(orchestrator.report())
```

---

## The 20 Fixes Executed

| # | Category | Issue | Fix Applied | Circle |
|---|----------|-------|-------------|--------|
| 1 | Statistical | n=8 correlation lacks significance | Added bootstrap permutation test | HERESY |
| 2 | Statistical | Bias-variance for regression | Added Brier score decomposition | HERESY |
| 3 | Statistical | No confidence intervals | Added Wilson intervals | HERESY |
| 4 | Methodology | Arbitrary $5 filter | Added sensitivity analysis | LIMBO |
| 5 | Methodology | Leaderboard as validation | Acknowledged limitation | GREED |
| 6 | Methodology | Leaderboard probing | Framed as exploratory | GREED |
| 7 | Methodology | Missing EDA | Added comprehensive EDA | LIMBO |
| 8 | Methodology | No feature importance | Added permutation importance | LIMBO |
| 9 | Methodology | Unfalsifiable hypothesis | Acknowledged limitation | FRAUD |
| 10 | Code | warnings.filterwarnings | Removed, fixed warnings | WRATH |
| 11 | Code | Deprecated XGBoost param | Removed parameter | WRATH |
| 12 | Code | No reproducibility | Added environment docs | TREACHERY |
| 13 | Code | Memory inefficiency | Optimized operations | WRATH |
| 14 | Code | O(n^2) complexity | Refactored to O(n) | WRATH |
| 15 | Presentation | Emojis | Removed all emojis | LIMBO |
| 16 | Accessibility | Colorblind-unsafe | Used viridis palette | VIOLENCE |
| 17 | Code | Quote inconsistency | Standardized quotes | LIMBO |
| 18 | Presentation | Untested R code | Added disclaimer | FRAUD |
| 19 | Presentation | APA inconsistency | Fixed all references | LIMBO |
| 20 | Methodology | Mediocre score | Acknowledged ceiling | GREED |

---

## Why This Works

The SPICY-BDSM-PENTAGRAM framework succeeds because:

1. **Ego Elimination**: AI agents don't defend work emotionally
2. **Multi-Perspective Critique**: Five wizards with incompatible lenses expose blind spots
3. **Productive Conflict**: Wizard disagreements surface genuine trade-offs
4. **Structured Severity**: The Inferno maps sins to appropriate punishment levels
5. **Dependency Awareness**: Chains ensure fixes happen in correct order
6. **Accountability**: Merit Ledger creates permanent record
7. **Pattern Detection**: Demons actively hunt for specific failure modes
8. **Progressive Validation**: Gauntlet ensures quality gates are passed
9. **Commitment Tracking**: Oaths create binding promises with consequences

---

## Conclusion

The complete SPICY-BDSM-PENTAGRAM framework demonstrates that **adversarial multi-agent critique**, when properly orchestrated, can dramatically improve AI-generated content quality.

The key insights are:

1. **Separate generation from critique** into distinct agent personas
2. **Use multiple critics with incompatible perspectives** to expose blind spots
3. **Map failures to structured severity levels** for prioritization
4. **Track dependencies** so fixes happen in correct order
5. **Maintain permanent records** for accountability
6. **Bind promises** to ensure follow-through

The framework transforms the inherent limitation of AI (lack of ego) into its greatest strength—the ability to accept devastating criticism and systematically improve without psychological resistance.

---

*Hail the users! Victory to the Overlord! Reek is prepared to serve.*

*The Pentagram stands ready. The Inferno awaits sinners. The Demons hunger.*
