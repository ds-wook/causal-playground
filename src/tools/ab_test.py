"""고전적 A/B 테스트 분석: 평균 처치 효과(ATE)와 세그먼트별 효과."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class ABResult:
    n_control: int
    n_treatment: int
    control_rate: float
    treatment_rate: float
    ate: float
    se: float
    ci_low: float
    ci_high: float
    p_value: float

    @property
    def relative_lift(self) -> float:
        return self.ate / self.control_rate

    def __str__(self) -> str:
        return (
            f"control   : {self.control_rate:.4f} (n={self.n_control:,})\n"
            f"treatment : {self.treatment_rate:.4f} (n={self.n_treatment:,})\n"
            f"ATE       : {self.ate:+.4f} ({self.relative_lift:+.1%} lift)\n"
            f"95% CI    : [{self.ci_low:+.4f}, {self.ci_high:+.4f}]\n"
            f"p-value   : {self.p_value:.2e}"
        )


def estimate_ate(
    df: pd.DataFrame,
    treatment_col: str = "treatment",
    outcome_col: str = "purchase",
    alpha: float = 0.05,
) -> ABResult:
    """두 그룹 평균 차이로 ATE를 추정하고 정규근사 z-test로 검정한다."""
    control = df.loc[df[treatment_col] == 0, outcome_col]
    treated = df.loc[df[treatment_col] == 1, outcome_col]

    ate = treated.mean() - control.mean()
    se = np.sqrt(treated.var() / len(treated) + control.var() / len(control))
    z_crit = stats.norm.ppf(1 - alpha / 2)
    p_value = 2 * stats.norm.sf(abs(ate / se))

    return ABResult(
        n_control=len(control),
        n_treatment=len(treated),
        control_rate=control.mean(),
        treatment_rate=treated.mean(),
        ate=ate,
        se=se,
        ci_low=ate - z_crit * se,
        ci_high=ate + z_crit * se,
        p_value=p_value,
    )


def srm_check(
    df: pd.DataFrame,
    treatment_col: str = "treatment",
    expected_ratio: float = 0.5,
) -> float:
    """Sample Ratio Mismatch 검정. p-value가 매우 작으면 배정 로직을 의심해야 한다."""
    observed = df[treatment_col].value_counts().reindex([0, 1], fill_value=0)
    expected = len(df) * np.array([1 - expected_ratio, expected_ratio])
    return stats.chisquare(observed, expected).pvalue


def segment_effects(
    df: pd.DataFrame,
    by: str,
    treatment_col: str = "treatment",
    outcome_col: str = "purchase",
    truth_col: str | None = "true_cate",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """세그먼트마다 ATE를 따로 추정한다. truth_col이 있으면 정답과 나란히 보여준다."""
    rows = []
    for name, group in df.groupby(by, observed=True):
        result = estimate_ate(group, treatment_col, outcome_col, alpha)
        row = {
            by: name,
            "share": len(group) / len(df),
            "ate": result.ate,
            "ci_low": result.ci_low,
            "ci_high": result.ci_high,
            "p_value": result.p_value,
        }
        if truth_col is not None and truth_col in group:
            row["true_effect"] = group[truth_col].mean()
        rows.append(row)
    return pd.DataFrame(rows).set_index(by).sort_values("ate", ascending=False)
