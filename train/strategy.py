"""Custom FL strategies wrapper.

This module provides:
  * FedAvgStrategy    – thin wrapper around Flower's FedAvg (for consistency)
  * FedProxStrategy   – same FedAvg aggregation but passes `mu` to clients
  * FedBNStrategy     – FedAvg aggregation **excluding BatchNorm params**
  * get_strategy(cfg) – helper that returns the correct strategy based on YAML cfg

Note: For MOON or other custom strategies you can extend this file similarly.
"""

from __future__ import annotations

from typing import List, Tuple, Dict

import flwr as fl
from flwr.common import Parameters, ndarrays_to_parameters, parameters_to_ndarrays
from omegaconf import DictConfig

# ──────────────────────────────────────────────────────────────
# FedAvg (thin wrapper)                                         
# ──────────────────────────────────────────────────────────────

class FedAvgStrategy(fl.server.strategy.FedAvg):
    """Just expose FedAvg directly for uniform API."""

    def __init__(self, cfg: DictConfig):
        super().__init__(min_fit_clients=cfg.fl.min_fit_clients,
                         min_available_clients=cfg.fl.min_available_clients)

# ──────────────────────────────────────────────────────────────
# FedProx Strategy                                             
# (server aggregation identical to FedAvg; passes `mu` to client)
# ──────────────────────────────────────────────────────────────

class FedProxStrategy(fl.server.strategy.FedAvg):
    """FedProx – server side same as FedAvg; mu passed via FitIns.config."""

    def __init__(self, cfg: DictConfig):
        self.mu: float = float(cfg.train.mu)
        super().__init__(min_fit_clients=cfg.fl.min_fit_clients,
                         min_available_clients=cfg.fl.min_available_clients)

    # Inject mu in Fit instructions
    def configure_fit(self, rnd, parameters, client_manager):  # noqa: D401
        fit_cfg = super().configure_fit(rnd, parameters, client_manager)
        new_cfg = []
        for client, fit_ins in fit_cfg:  # type: ignore[assignment]
            if fit_ins.config is None:
                fit_ins.config = {}
            fit_ins.config["mu"] = self.mu
            new_cfg.append((client, fit_ins))
        return new_cfg

# ──────────────────────────────────────────────────────────────
# FedBN Strategy                                               
#   – BN affine & running stats are kept local (no aggregation)
# ──────────────────────────────────────────────────────────────

def _is_bn_param(name: str) -> bool:
    bn_keywords = ["bn", "running_mean", "running_var"]
    return any(kw in name.lower() for kw in bn_keywords)

class FedBNStrategy(fl.server.strategy.FedAvg):
    """Implements FedBN aggregation: exclude BN params from averaging."""

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException] | None,
    ) -> Tuple[Parameters | None, Dict[str, fl.common.Scalar]]:
        if not results:
            return None, {}

        # Convert each client's ndarray list to state_dict-like list-of-arrays
        params_ndarrays = [parameters_to_ndarrays(res.parameters) for _, res in results]

        # Use first client's params as base
        base = params_ndarrays[0]
        n_clients = len(params_ndarrays)

        # Average non-BN params
        for i, name in enumerate(self._parameter_names):  # type: ignore[attr-defined]
            if _is_bn_param(name):
                continue  # keep local BN as-is (use base)
            # compute mean across clients
            stacked = [client_params[i] for client_params in params_ndarrays]
            base[i] = sum(stacked) / n_clients

        new_params = ndarrays_to_parameters(base)
        return new_params, {}

    # Keep track of param order names – call once when first parameters seen
    _parameter_names: List[str] | None = None  # type: ignore[assignment]

    def initialize_parameters(self, client_manager):  # noqa: D401
        params, _ = super().initialize_parameters(client_manager)
        if params is not None and self._parameter_names is None:
            self._parameter_names = [k for k in self._params_to_state_dict(params).keys()]  # type: ignore[arg-type]
        return params

    def _params_to_state_dict(self, parameters: Parameters):
        """Convert Parameters to an ordered dict with names (helper)."""
        # Hack: we rely on each ndarray order is consistent across clients
        # Flower does not carry param names, so we store on first call from Torch model
        import torch
        from torch import nn
        tmp_model = nn.Module()  # placeholder (unused), we only need state_dict keys
        state_dict = tmp_model.state_dict()  # empty OrderedDict
        ndarrays = parameters_to_ndarrays(parameters)
        for key, array in zip(state_dict.keys(), ndarrays):
            state_dict[key] = torch.tensor(array)
        return state_dict

# ──────────────────────────────────────────────────────────────
# Factory helper                                                
# ──────────────────────────────────────────────────────────────

def get_strategy(cfg: DictConfig) -> fl.server.strategy.Strategy:
    strat = cfg.train.strategy.lower()
    if strat == "fedavg":
        return FedAvgStrategy(cfg)
    if strat == "fedprox":
        return FedProxStrategy(cfg)
    if strat == "fedbn":
        return FedBNStrategy(cfg)
    raise ValueError(f"Unknown strategy: {cfg.train.strategy}")
