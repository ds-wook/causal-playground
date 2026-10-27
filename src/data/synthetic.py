"""이커머스 쿠폰 실험 합성 데이터.

실제 데이터와 달리 유저별 진짜 처치 효과(true_cate)를 알고 있으므로
A/B 분석이나 uplift 모델이 얼마나 정확한지 채점할 수 있다.
"""

from collections.abc import Mapping

import numpy as np
import pandas as pd


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def _standardize(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / x.std()


def make_coupon_data(
    segment_effects: Mapping[str, float],
    n: int = 50_000,
    treatment_ratio: float = 0.5,
    confounded: bool = False,
    seed: int = 42,
) -> pd.DataFrame:
    """쿠폰 지급(treatment) → 7일 내 구매(purchase) 데이터를 생성한다.

    segment_effects는 세그먼트별 진짜 쿠폰 효과(구매 확률의 절대 변화량)로,
    persuadable / sure_thing / sleeping_dog / other 키를 모두 가져야 한다.
    confounded=False면 A/B 테스트처럼 랜덤 배정,
    True면 마케터가 활동 많은 유저에게 쿠폰을 몰아준 관측 데이터가 된다.
    """
    rng = np.random.default_rng(seed)

    visits_30d = rng.poisson(lam=3, size=n)
    past_spend = rng.lognormal(mean=11, sigma=1, size=n)
    tenure_days = np.clip(rng.exponential(scale=500, size=n), 1, 3650).astype(int)
    is_mobile = rng.binomial(1, 0.6, size=n)

    z_visits = _standardize(visits_30d.astype(float))
    z_spend = _standardize(np.log(past_spend))

    segment = np.select(
        [
            (tenure_days >= 730) & (visits_30d <= 2),
            past_spend >= np.quantile(past_spend, 0.8),
            (tenure_days < 180) & (visits_30d <= 3),
        ],
        ["sleeping_dog", "sure_thing", "persuadable"],
        default="other",
    )

    p_control = _sigmoid(-2.0 + 0.35 * z_visits + 0.5 * z_spend + 0.1 * is_mobile)
    tau = pd.Series(segment).map(dict(segment_effects)).to_numpy(dtype=float)
    p_treated = np.clip(p_control + tau, 0, 1)

    if confounded:
        propensity = _sigmoid(-0.5 + 1.0 * z_visits + 0.5 * z_spend)
    else:
        propensity = np.full(n, treatment_ratio)
    treatment = rng.binomial(1, propensity)

    purchase = rng.binomial(1, np.where(treatment == 1, p_treated, p_control))

    return pd.DataFrame(
        {
            "visits_30d": visits_30d,
            "past_spend": past_spend.round(-2),
            "tenure_days": tenure_days,
            "is_mobile": is_mobile,
            "treatment": treatment,
            "purchase": purchase,
            # 아래는 현실에서는 관측할 수 없는 정답 컬럼
            "segment": segment,
            "propensity": propensity,
            "true_cate": p_treated - p_control,
        }
    )
