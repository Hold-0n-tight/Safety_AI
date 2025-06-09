"""Create global evaluation function.

Optionally, also define a new Server class (please note this is not needed in most
settings).
"""

from collections import OrderedDict
from typing import Callable, Dict, Optional, Tuple, List

import torch
import flwr as fl
from flwr.common.typing import NDArrays, Scalar
from flwr.server.client_proxy import ClientProxy
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from .models import init_net, test


class MOONServer(fl.server.Server):
    """Custom server implementation for MOON."""

    def __init__(
        self,
        *,
        client_manager: fl.server.ClientManager,
        strategy: fl.server.strategy.Strategy,
    ):
        super().__init__(client_manager=client_manager, strategy=strategy)

    def fit_clients(
        self,
        num_clients: int,
        timeout: Optional[float],
        fit_ins: fl.common.FitIns,
    ) -> List[Tuple[ClientProxy, fl.common.FitRes]]:
        """Add round number to client config."""
        # Get current round number
        current_round = self.strategy.current_round
        
        # Add round number to config
        if fit_ins.config is None:
            fit_ins.config = {}
        fit_ins.config["current_round"] = current_round
        
        # Call parent class's fit_clients
        return super().fit_clients(num_clients, timeout, fit_ins)


def gen_evaluate_fn(
    testloader: DataLoader,
    device: torch.device,
    cfg: DictConfig,
) -> Callable[
    [int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]
]:
    """Generate the function for centralized evaluation."""

    def evaluate(
        server_round: int, parameters_ndarrays: NDArrays, config: Dict[str, Scalar]
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        # pylint: disable=unused-argument
        net = init_net(cfg.dataset.name, cfg.model.name, cfg.model.output_dim)
        params_dict = zip(net.state_dict().keys(), parameters_ndarrays)
        state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)
        net.to(device)

        accuracy, loss = test(net, testloader, device=device)
        return loss, {"accuracy": accuracy}

    return evaluate
