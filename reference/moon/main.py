"""Create and connect the building blocks for your experiments; start the simulation.

It includes processioning the dataset, instantiate strategy, specify how the global
model is going to be evaluated, etc. At the end, this script saves the results.
"""

import os
import random
import shutil
from pathlib import Path

# these are the basic packages you'll need here
# feel free to remove some if aren't needed
import flwr as fl
import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

# 상대 경로로 import 수정
from .client import gen_client_fn
from .dataset import get_dataloader
from .dataset_preparation import partition_data
from .utils import plot_metric_from_history, save_comparison_results, plot_comparison_results
from .server import gen_evaluate_fn, MOONServer


@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run the baseline.

    Parameters
    ----------
    cfg : DictConfig
        An omegaconf object that stores the hydra config.
    """
    # Clean the model directory to save models for MOON
    if cfg.alg == "moon":
        if os.path.exists(cfg.model.dir):
            shutil.rmtree(cfg.model.dir)
    # 1. Print parsed config
    print(OmegaConf.to_yaml(cfg))

    # 2. Prepare your dataset
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    (
        _,
        _,
        _,
        _,
        net_dataidx_map,
    ) = partition_data(
        dataset=cfg.dataset.name,
        datadir=cfg.dataset.dir,
        partition=cfg.dataset.partition,
        num_clients=cfg.num_clients,
        beta=cfg.dataset.beta,
    )

    _, test_global_dl, _, _ = get_dataloader(
        dataset=cfg.dataset.name,
        datadir=cfg.dataset.dir,
        train_bs=cfg.batch_size,
        test_bs=32,
    )

    trainloaders = []
    testloaders = []
    for idx in range(cfg.num_clients):
        train_dl, test_dl, _, _ = get_dataloader(
            cfg.dataset.name, cfg.dataset.dir, cfg.batch_size, 32, net_dataidx_map[idx]
        )

        trainloaders.append(train_dl)
        testloaders.append(test_dl)
    # 3. Define your clients
    # Define a function that returns another function that will be used during
    # simulation to instantiate each individual client
    client_fn = gen_client_fn(
        trainloaders=trainloaders,
        testloaders=testloaders,
        cfg=cfg,
    )

    # get function that will executed by the strategy's evaluate() method
    # Set server's device
    device = (
        torch.device("cuda:0")
        if torch.cuda.is_available() and cfg.server_device == "cuda"
        else "cpu"
    )
    evaluate_fn = gen_evaluate_fn(test_global_dl, device=device, cfg=cfg)

    # 4. Define your strategy
    strategy = fl.server.strategy.FedAvg(
        # Clients in MOON do not perform federated evaluation
        # (see the client's evaluate())
        fraction_fit=cfg.fraction_fit,
        fraction_evaluate=0.0,
        evaluate_fn=evaluate_fn,
    )
    
    # 5. Start Simulation with custom server
    # Create a client manager
    client_manager = fl.server.SimpleClientManager()
    
    # Create custom server instance
    server = MOONServer(
        client_manager=client_manager,
        strategy=strategy,
    )
    
    # Start simulation with custom server
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        client_resources={
            "num_cpus": cfg.client_resources.num_cpus,
            "num_gpus": cfg.client_resources.num_gpus,
        },
        strategy=strategy,
        server=server,  # Use our custom server
    )
    # remove saved models
    if cfg.alg == "moon":
        shutil.rmtree(cfg.model.dir)

    # 6. Save your results
    # Experiment completed. Now we save the results and
    # generate plots using the `history`
    print("................")
    print(history)

    # Hydra automatically creates an output directory
    # Let's retrieve it and save some results there
    save_path = HydraConfig.get().runtime.output_dir

    # plot results and include them in the readme
    strategy_name = strategy.__class__.__name__
    file_suffix: str = (
        f"_{strategy_name}"
        f"{'_dataset' if cfg.dataset.name else ''}"
        f"_C={cfg.num_clients}"
        f"_B={cfg.batch_size}"
        f"_E={cfg.num_epochs}"
        f"_R={cfg.num_rounds}"
        f"_mu={cfg.mu}"
    )

    plot_metric_from_history(
        history,
        Path(save_path),
        (file_suffix),
    )
    
    # Check if we need to compare with base MOON results
    if hasattr(cfg, 'compare_with_base') and cfg.compare_with_base:
        # Get the base MOON results path
        base_results_path = cfg.base_results_path
        
        # Check if base results exist
        if os.path.exists(base_results_path):
            # Load base results
            import json
            with open(base_results_path, 'r') as f:
                base_results = json.load(f)
            
            # Extract metrics from current history
            current_results = {
                "accuracy": history.metrics_centralized["accuracy"][-1][1],
                "loss": history.metrics_centralized["loss"][-1][1] if "loss" in history.metrics_centralized else None,
                "config": OmegaConf.to_container(cfg, resolve=True)
            }
            
            # Save comparison results
            save_comparison_results(
                base_results=base_results,
                bn_drift_results=current_results,
                save_path=Path(save_path),
                experiment_name=f"moon_comparison{file_suffix}"
            )
            
            # Plot comparison results
            # Load base history
            import pickle
            with open(os.path.join(os.path.dirname(base_results_path), 'history.pkl'), 'rb') as f:
                base_history = pickle.load(f)
            
            plot_comparison_results(
                base_history=base_history,
                bn_drift_history=history,
                save_plot_path=Path(save_path),
                experiment_name=f"moon_comparison{file_suffix}"
            )
            
            print(f"Comparison with base MOON results completed and saved to {save_path}")
        else:
            print(f"Base results path {base_results_path} does not exist. Skipping comparison.")
    
    # Save current results for future comparison
    if hasattr(cfg, 'save_for_comparison') and cfg.save_for_comparison:
        # Create results directory if it doesn't exist
        results_dir = Path(cfg.results_dir)
        os.makedirs(results_dir, exist_ok=True)
        
        # Save current results
        import json
        current_results = {
            "accuracy": history.metrics_centralized["accuracy"][-1][1],
            "loss": history.metrics_centralized["loss"][-1][1] if "loss" in history.metrics_centralized else None,
            "config": OmegaConf.to_container(cfg, resolve=True)
        }
        
        with open(results_dir / f"results{file_suffix}.json", 'w') as f:
            json.dump(current_results, f, indent=4)
        
        # Save history
        import pickle
        with open(results_dir / f"history{file_suffix}.pkl", 'wb') as f:
            pickle.dump(history, f)
        
        print(f"Results saved for future comparison to {results_dir}")


if __name__ == "__main__":
    main()
