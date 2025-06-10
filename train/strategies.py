"""
strategies.py

Custom FL strategies for Flower.

* FedAvgStrategy  – Wrapper form. Use base FedAvg
* FedProxStrategy – Send client to FedAvg + μ(proximal)
* FedBNStrategy   – Exclude BatchNorm parameter in average (FedBN)
* get_strategy()  – Returns appropriate strategy base on .yaml configuration
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import flwr as fl
import numpy as np
from omegaconf import DictConfig

from models import init_net


# ──────────────────────────────────────────────────────────────
# Helper: BN 여부 판별
# ──────────────────────────────────────────────────────────────
def _is_bn_param(name: str) -> bool:
    return (
        ".running_mean" in name
        or ".running_var" in name
        or ".num_batches_tracked" in name
    )


# ──────────────────────────────────────────────────────────────
# 1. FedAvg (래퍼)
# ──────────────────────────────────────────────────────────────
class FedAvgStrategy(fl.server.strategy.FedAvg):
    """얇은 래퍼—Flower 기본 FedAvg와 동일하지만 cfg 인자를 통일."""

    def __init__(self, cfg: DictConfig):
        super().__init__(
            min_fit_clients=cfg.fl.min_fit_clients,
            min_available_clients=cfg.fl.min_available_clients,
            fraction_fit=cfg.fl.get("fraction_fit", 1.0),
        )


# ──────────────────────────────────────────────────────────────
# 2. FedProx
# ──────────────────────────────────────────────────────────────
class FedProxStrategy(fl.server.strategy.FedAvg):
    """FedAvg + proximal term(μ)을 클라이언트 config로 전달."""

    def __init__(self, cfg: DictConfig):
        self.mu: float = float(cfg.train.mu)
        super().__init__(
            min_fit_clients=cfg.fl.min_fit_clients,
            min_available_clients=cfg.fl.min_available_clients,
            fraction_fit=cfg.fl.get("fraction_fit", 1.0),
        )

    def configure_fit(  # noqa: D401
        self,
        rnd: int,
        parameters: fl.common.Parameters,
        client_manager: fl.server.client_manager.ClientManager,
    ) -> List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitIns]]:
        # 기본 FedAvg 설정을 가져온 뒤 config에 μ 추가
        fit_config = super().configure_fit(rnd, parameters, client_manager)
        patched: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitIns]] = []
        for client_proxy, fit_ins in fit_config:
            new_conf = dict(fit_ins.config)
            new_conf["mu"] = self.mu
            patched.append((client_proxy, fl.common.FitIns(fit_ins.parameters, new_conf)))
        return patched


# ──────────────────────────────────────────────────────────────
# 3. FedBN  (BN 파라미터 제외 평균)
# ──────────────────────────────────────────────────────────────
class FedBNStrategy(fl.server.strategy.FedAvg):
    """BatchNorm 파라미터를 평균에서 제외하는 FedBN 구현."""

    def __init__(self, cfg: DictConfig):
        # 모델 한 번 생성 → state_dict 키 순서 확보
        model = init_net(cfg.model.name, cfg.model.output_dim)
        self._parameter_names: List[str] = list(model.state_dict().keys())

        super().__init__(
            min_fit_clients=cfg.fl.min_fit_clients,
            min_available_clients=cfg.fl.min_available_clients,
            fraction_fit=cfg.fl.get("fraction_fit", 1.0),
        )

    def aggregate_fit(  # noqa: D401
        self,
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures,
    ) -> Tuple[fl.common.Parameters | None, Dict[str, fl.common.Scalar]]:
        # 기본 FedAvg 결과
        agg_params, metrics = super().aggregate_fit(rnd, results, failures)
        if agg_params is None:
            return None, metrics

        # ndarrays로 변환
        agg_ndarrays = fl.common.parameters_to_ndarrays(agg_params)
        first_ndarrays = fl.common.parameters_to_ndarrays(results[0][1].parameters)

        # BN 파라미터는 첫 클라이언트 값 그대로 사용
        merged: List[np.ndarray] = []
        for name, w_avg, w_first in zip(self._parameter_names, agg_ndarrays, first_ndarrays):
            merged.append(w_first if _is_bn_param(name) else w_avg)

        return fl.common.ndarrays_to_parameters(merged), metrics


# ──────────────────────────────────────────────────────────────
# 4. Strategy Factory
# ──────────────────────────────────────────────────────────────
def get_strategy(cfg: DictConfig) -> fl.server.strategy.Strategy:
    """cfg.train.strategy 문자열에 맞는 Strategy 인스턴스를 반환."""
    strat = cfg.train.strategy.lower()
    if strat == "fedavg":
        return FedAvgStrategy(cfg)
    if strat == "fedprox":
        return FedProxStrategy(cfg)
    if strat == "fedbn":
        return FedBNStrategy(cfg)
    raise ValueError(f"Unknown strategy '{cfg.train.strategy}'")
