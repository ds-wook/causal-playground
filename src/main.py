import json
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

from data.synthetic import make_coupon_data
from models.uplift import cate_by_segment, estimate_cate, evaluate_cate
from tools.ab_test import estimate_ate, segment_effects, srm_check


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    pd.set_option("display.float_format", "{:+.4f}".format)
    df = make_coupon_data(**cfg.data)
    ab = cfg.analysis
    cols = {"treatment_col": ab.treatment_col, "outcome_col": ab.outcome_col}

    print("1. A/B 테스트: 쿠폰이 평균적으로 효과가 있나?")
    print(
        f"SRM p-value: {srm_check(df, ab.treatment_col, cfg.data.treatment_ratio):.3f}"
    )
    print(estimate_ate(df, **cols, alpha=ab.alpha))
    print(f"(정답 ATE: {df['true_cate'].mean():+.4f})")

    print("2. 한계: 평균 뒤에 숨은 이질성")
    print("관측 가능한 피처로 쪼개 보기 (가입 기간)")
    df["tenure_bucket"] = pd.cut(
        df["tenure_days"],
        bins=list(ab.tenure_bins),
        labels=list(ab.tenure_labels),
    )
    print(segment_effects(df, by="tenure_bucket", **cols, alpha=ab.alpha))

    print("\n정답 세그먼트로 쪼개 보기 (현실에서는 알 수 없음)")
    print(segment_effects(df, by="segment", **cols, alpha=ab.alpha))

    print(f"\n3. Uplift 모델 ({cfg.model.name}): 누구에게 쿠폰이 효과 있나?")
    train, test = train_test_split(
        df,
        test_size=cfg.uplift.test_size,
        random_state=cfg.uplift.seed,
        stratify=df[ab.treatment_col],
    )
    features = list(cfg.uplift.features)
    # A/B 테스트는 배정 확률을 설계로 알고 있다. 관측 데이터는 모델이 추정해야 한다.
    if cfg.data.confounded:
        p_train = p_test = None
    else:
        p_train = np.full(len(train), cfg.data.treatment_ratio)
        p_test = np.full(len(test), cfg.data.treatment_ratio)

    pred = estimate_cate(
        instantiate(cfg.model.estimator),
        train[features].to_numpy(),
        train[ab.treatment_col].to_numpy(),
        train[ab.outcome_col].to_numpy(),
        test[features].to_numpy(),
        p_train=p_train,
        p_test=p_test,
    )
    truth = test["true_cate"].to_numpy()

    metrics = evaluate_cate(pred, truth)
    for key, value in metrics.items():
        print(f"{key:<9}: {value:+.4f}")
    print("\n세그먼트별 예측 vs 정답 (테스트셋)")
    print(cate_by_segment(pred, truth, test["segment"]))

    output_dir = Path(HydraConfig.get().runtime.output_dir)
    (output_dir / "metrics.json").write_text(
        json.dumps({"model": cfg.model.name, **metrics}, indent=2)
    )
    test[["segment", "true_cate"]].assign(pred_cate=pred).to_csv(
        output_dir / "cate.csv"
    )


if __name__ == "__main__":
    main()
