"""causalml meta-learner로 개인별 처치 효과(CATE)를 추정하고 정답과 비교한다."""

import inspect

import numpy as np
import pandas as pd
from scipy import stats


def _call(method, *args, **kwargs):
    """meta-learner마다 받는 인자가 조금씩 달라서, 지원하는 kwargs만 넘긴다."""
    params = inspect.signature(method).parameters
    return method(*args, **{k: v for k, v in kwargs.items() if k in params})


def estimate_cate(
    estimator,
    X_train: np.ndarray,
    t_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    p_train: np.ndarray | None = None,
    p_test: np.ndarray | None = None,
) -> np.ndarray:
    """학습 데이터로 fit 하고 테스트 유저의 CATE를 예측한다.

    p(propensity)가 None이면 X/R-learner는 내부에서 직접 추정한다.
    """
    _call(estimator.fit, X_train, t_train, y_train, p=p_train, verbose=False)
    cate = _call(estimator.predict, X_test, p=p_test, verbose=False)
    return np.asarray(cate).reshape(-1)


def evaluate_cate(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """정답 CATE 대비 정확도. 타게팅에는 크기보다 순위(spearman)가 더 중요하다."""
    return {
        "pred_ate": float(pred.mean()),
        "true_ate": float(truth.mean()),
        "rmse": float(np.sqrt(np.mean((pred - truth) ** 2))),
        "pearson": float(stats.pearsonr(pred, truth).statistic),
        "spearman": float(stats.spearmanr(pred, truth).statistic),
    }


def cate_by_segment(
    pred: np.ndarray,
    truth: np.ndarray,
    segment: pd.Series,
) -> pd.DataFrame:
    """세그먼트별로 예측 CATE 평균과 정답 평균을 나란히 본다."""
    frame = pd.DataFrame(
        {"segment": segment.to_numpy(), "pred_cate": pred, "true_cate": truth}
    )
    return (
        frame.groupby("segment")
        .agg(
            share=("pred_cate", "size"),
            pred_cate=("pred_cate", "mean"),
            true_cate=("true_cate", "mean"),
        )
        .assign(share=lambda d: d["share"] / len(frame))
        .sort_values("true_cate", ascending=False)
    )
