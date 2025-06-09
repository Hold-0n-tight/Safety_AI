"""Define any utility function.

They are not directly relevant to  the other (more FL specific) python modules. For
example, you may define here things like: loading a model from a checkpoint, saving
results, plotting.
"""

from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr.server.history import History


def compute_accuracy(model, dataloader, device="cpu", multiloader=False):
    """Compute accuracy."""
    was_training = False
    if model.training:
        model.eval()
        was_training = True

    true_labels_list, pred_labels_list = np.array([]), np.array([])

    correct, total = 0, 0
    if device == "cpu":
        criterion = nn.CrossEntropyLoss()
    elif "cuda" in device.type:
        criterion = nn.CrossEntropyLoss().cuda()
    loss_collector = []
    if multiloader:
        for loader in dataloader:
            with torch.no_grad():
                for _, (x, target) in enumerate(loader):
                    if device != "cpu":
                        x, target = x.cuda(), target.to(dtype=torch.int64).cuda()
                    _, _, out = model(x)
                    if len(target) == 1:
                        loss = criterion(out, target)
                    else:
                        loss = criterion(out, target)
                    _, pred_label = torch.max(out.data, 1)
                    loss_collector.append(loss.item())
                    total += x.data.size()[0]
                    correct += (pred_label == target.data).sum().item()

                    if device == "cpu":
                        pred_labels_list = np.append(
                            pred_labels_list, pred_label.numpy()
                        )
                        true_labels_list = np.append(
                            true_labels_list, target.data.numpy()
                        )
                    else:
                        pred_labels_list = np.append(
                            pred_labels_list, pred_label.cpu().numpy()
                        )
                        true_labels_list = np.append(
                            true_labels_list, target.data.cpu().numpy()
                        )
        avg_loss = sum(loss_collector) / len(loss_collector)
    else:
        with torch.no_grad():
            for _, (x, target) in enumerate(dataloader):
                # print("x:",x)
                if device != "cpu":
                    x, target = x.cuda(), target.to(dtype=torch.int64).cuda()
                _, _, out = model(x)
                loss = criterion(out, target)
                _, pred_label = torch.max(out.data, 1)
                loss_collector.append(loss.item())
                total += x.data.size()[0]
                correct += (pred_label == target.data).sum().item()

                if device == "cpu":
                    pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
                    true_labels_list = np.append(true_labels_list, target.data.numpy())
                else:
                    pred_labels_list = np.append(
                        pred_labels_list, pred_label.cpu().numpy()
                    )
                    true_labels_list = np.append(
                        true_labels_list, target.data.cpu().numpy()
                    )
            avg_loss = sum(loss_collector) / len(loss_collector)

    if was_training:
        model.train()

    return correct / float(total), avg_loss


def plot_metric_from_history(
    hist: History,
    save_plot_path: Path,
    suffix: Optional[str] = "",
) -> None:
    """Plot data from Flower server History.

    Parameters
    ----------
    hist : History
        Object containing evaluation for all rounds.
    save_plot_path : Path
        Folder to save the plot to.
    suffix: Optional[str]
        Optional string to add at the end of the filename for the plot.
    """
    metric_type = "centralized"
    metric_dict = (
        hist.metrics_centralized
        if metric_type == "centralized"
        else hist.metrics_distributed
    )
    rounds, values = zip(*metric_dict["accuracy"])

    # Plot the curve
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, values)
    plt.xlabel("#round")
    plt.ylabel("Test accuracy")
    plt.legend()
    plt.show()

    plt.savefig(Path(save_plot_path) / Path(f"{metric_type}_metrics{suffix}.png"))
    plt.close()


def im2col(input_tensor, kernel_size, stride=1, padding=0):
    """Convert input tensor to column matrix for convolution operation.
    GPU optimized version.
    """
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    
    # Get input dimensions
    B, C, H, W = input_tensor.shape
    k_h, k_w = kernel_size
    s_h, s_w = stride
    p_h, p_w = padding
    
    # Calculate output dimensions
    H_out = (H + 2 * p_h - k_h) // s_h + 1
    W_out = (W + 2 * p_w - k_w) // s_w + 1
    
    # Pad input tensor if needed (on GPU)
    if p_h > 0 or p_w > 0:
        input_tensor = F.pad(input_tensor, (p_w, p_w, p_h, p_h), mode='constant', value=0)
    
    # Create output tensor on the same device as input
    col = torch.zeros((B, C, k_h, k_w, H_out, W_out), device=input_tensor.device)
    
    # Fill output tensor
    for y in range(k_h):
        y_max = y + s_h * H_out
        for x in range(k_w):
            x_max = x + s_w * W_out
            col[:, :, y, x, :, :] = input_tensor[:, :, y:y_max:s_h, x:x_max:s_w]
    
    # Reshape output tensor
    col = col.permute(1, 2, 3, 0, 4, 5).contiguous()
    col = col.view(C * k_h * k_w, B * H_out * W_out)
    
    return col


def compute_bn_spectrum_similarity(model: nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device) -> float:
    """
    Compute similarity between input and output singular value spectrums of BN layers.
    Uses im2col for convolutional layers and directly computes SVD.
    Only uses top 20 singular values for similarity computation.
    Optimized for GPU computation and reduced computation by sampling.
    """
    model.eval()
    similarities = []
    bn_conv_pairs = {}
    TOP_K = 20  # Number of top singular values to use

    # First, identify CNN layers that precede BN layers
    prev_layer = None
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d):
            prev_layer = layer
        elif isinstance(layer, nn.BatchNorm2d) and prev_layer is not None:
            bn_conv_pairs[layer] = prev_layer
            prev_layer = None

    if not bn_conv_pairs:
        raise ValueError("No valid BN-Conv pairs found in the model")

    # Register hooks for BN layers
    bn_inputs = {}
    bn_outputs = {}
    hooks = []

    def forward_pre_hook(module, input):
        bn_inputs[module] = input[0].detach()
        return input

    def forward_hook(module, input, output):
        bn_outputs[module] = output.detach()

    for bn_layer in bn_conv_pairs.keys():
        hooks.append(bn_layer.register_forward_hook(forward_hook))
        hooks.append(bn_layer.register_forward_pre_hook(forward_pre_hook))

    try:
        # Process a single batch of data
        data, _ = next(iter(dataloader))
        data = data.to(device)
        model(data)

        # Compute similarity for each BN layer
        for bn_idx, (bn_layer, conv_layer) in enumerate(bn_conv_pairs.items()):
            if bn_layer not in bn_inputs or bn_layer not in bn_outputs:
                continue

            input_features = bn_inputs[bn_layer]
            output_features = bn_outputs[bn_layer]

            # Randomly select a single sample from the batch
            sample_idx = torch.randint(0, input_features.size(0), (1,)).item()
            input_features = input_features[sample_idx:sample_idx+1]
            output_features = output_features[sample_idx:sample_idx+1]

            # Get conv layer parameters
            kernel_size = conv_layer.kernel_size[0]
            stride = conv_layer.stride[0]
            padding = conv_layer.padding[0]

            # Transform using im2col (all operations on GPU)
            input_col = im2col(input_features, kernel_size, stride, padding)
            output_col = im2col(output_features, kernel_size, stride, padding)

            # Normalize matrices for numerical stability
            input_norm = torch.norm(input_col, p='fro')
            output_norm = torch.norm(output_col, p='fro')
            
            if input_norm == 0 or output_norm == 0:
                raise ValueError(f"Zero norm detected in BN layer {bn_idx + 1}")
                
            input_col = input_col / input_norm
            output_col = output_col / output_norm

            # Compute SVD with gesvd driver for better stability
            try:
                input_s = torch.linalg.svdvals(input_col, driver='gesvd')
                output_s = torch.linalg.svdvals(output_col, driver='gesvd')
            except RuntimeError as e:
                print(f"SVD computation failed for BN layer {bn_idx + 1}: {str(e)}")
                raise e

            # Get top k singular values and normalize them
            k = min(TOP_K, min(len(input_s), len(output_s)))
            input_s = input_s[:k]
            output_s = output_s[:k]

            # Normalize singular values
            input_s = input_s / torch.norm(input_s)
            output_s = output_s / torch.norm(output_s)

            if len(input_s) == 0 or len(output_s) == 0:
                raise ValueError(f"No singular values found in BN layer {bn_idx + 1}")

            # Compute cosine similarity on GPU using top k values
            similarity = F.cosine_similarity(input_s.unsqueeze(0), output_s.unsqueeze(0), dim=1).item()
            
            # Clip similarity to [-1, 1]
            similarity = min(max(similarity, -1.0), 1.0)
            similarities.append(similarity)

    finally:
        # Remove hooks
        for hook in hooks:
            hook.remove()

    if not similarities:
        raise ValueError("No similarities computed")
    
    return min(similarities)


def compute_adaptive_mu(model: nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device, mu_base: float, cfg=None) -> float:
    """
    Compute adaptive mu based on BN spectrum similarity.
    Following the formula: μₖ = μ_base * (1 - ρₖ⁸)
    where ρₖ is the BN spectrum similarity.
    """
    if not cfg or not hasattr(cfg, 'bn_drift_control') or not cfg.bn_drift_control.enabled:
        return mu_base
    
    similarity = compute_bn_spectrum_similarity(model, dataloader, device)
    adaptive_factor = 1 - similarity ** 8
    adaptive_mu = mu_base * adaptive_factor
    
    if torch.isnan(torch.tensor(adaptive_mu)) or adaptive_mu < 0:
        raise ValueError(f"Invalid adaptive_mu computed: {adaptive_mu}")
        
    print(f"BN Drift Control: ρₖ={similarity:.4f}, ρₖ⁸={similarity**8:.4f}, μₖ={adaptive_mu:.4f} (base μ={mu_base:.4f})")
    return adaptive_mu


def save_comparison_results(
    base_results: Dict[str, Any],
    bn_drift_results: Dict[str, Any],
    save_path: Path,
    experiment_name: str = "moon_comparison"
) -> None:
    """Save comparison results between base MOON and BN drift control MOON.
    
    Parameters
    ----------
    base_results : Dict[str, Any]
        Results from base MOON experiment
    bn_drift_results : Dict[str, Any]
        Results from BN drift control MOON experiment
    save_path : Path
        Path to save the comparison results
    experiment_name : str, optional
        Name of the experiment, by default "moon_comparison"
    """
    import pandas as pd
    import os
    from datetime import datetime
    
    # Create save directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Create timestamp for unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Extract metrics from results
    metrics = {}
    
    # Accuracy metrics
    if "accuracy" in base_results and "accuracy" in bn_drift_results:
        base_acc = base_results["accuracy"]
        bn_drift_acc = bn_drift_results["accuracy"]
        metrics["Accuracy"] = {
            "Base MOON": base_acc,
            "BN Drift Control MOON": bn_drift_acc,
            "Improvement (%)": ((bn_drift_acc - base_acc) / base_acc * 100) if base_acc > 0 else 0
        }
    
    # Loss metrics
    if "loss" in base_results and "loss" in bn_drift_results:
        base_loss = base_results["loss"]
        bn_drift_loss = bn_drift_results["loss"]
        metrics["Loss"] = {
            "Base MOON": base_loss,
            "BN Drift Control MOON": bn_drift_loss,
            "Improvement (%)": ((base_loss - bn_drift_loss) / base_loss * 100) if base_loss > 0 else 0
        }
    
    # Personalization score metrics (if available)
    if "personalization_score" in base_results and "personalization_score" in bn_drift_results:
        base_pers = base_results["personalization_score"]
        bn_drift_pers = bn_drift_results["personalization_score"]
        metrics["Personalization Score"] = {
            "Base MOON": base_pers,
            "BN Drift Control MOON": bn_drift_pers,
            "Improvement (%)": ((bn_drift_pers - base_pers) / base_pers * 100) if base_pers > 0 else 0
        }
    
    # BN drift metrics (if available)
    if "bn_drift" in bn_drift_results:
        metrics["BN Drift"] = {
            "Base MOON": base_results.get("bn_drift", "N/A"),
            "BN Drift Control MOON": bn_drift_results["bn_drift"],
            "Improvement (%)": "N/A"
        }
    
    # Create DataFrame
    df = pd.DataFrame(metrics).T
    
    # Save to CSV
    csv_path = save_path / f"{experiment_name}_{timestamp}.csv"
    df.to_csv(csv_path)
    
    # Save to Excel
    excel_path = save_path / f"{experiment_name}_{timestamp}.xlsx"
    df.to_excel(excel_path)
    
    # Create a more detailed report
    report_path = save_path / f"{experiment_name}_{timestamp}_report.txt"
    with open(report_path, "w") as f:
        f.write(f"Comparison Report: {experiment_name}\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("Experiment Configuration:\n")
        f.write("-" * 50 + "\n")
        if "config" in base_results:
            f.write("Base MOON Configuration:\n")
            for key, value in base_results["config"].items():
                f.write(f"  {key}: {value}\n")
        if "config" in bn_drift_results:
            f.write("\nBN Drift Control MOON Configuration:\n")
            for key, value in bn_drift_results["config"].items():
                f.write(f"  {key}: {value}\n")
        
        f.write("\nResults Comparison:\n")
        f.write("-" * 50 + "\n")
        for metric, values in metrics.items():
            f.write(f"{metric}:\n")
            f.write(f"  Base MOON: {values['Base MOON']}\n")
            f.write(f"  BN Drift Control MOON: {values['BN Drift Control MOON']}\n")
            if values['Improvement (%)'] != "N/A":
                f.write(f"  Improvement: {values['Improvement (%)']:.2f}%\n")
            f.write("\n")
        
        f.write("Conclusion:\n")
        f.write("-" * 50 + "\n")
        if "accuracy" in metrics:
            if metrics["Accuracy"]["Improvement (%)"] > 0:
                f.write(f"BN Drift Control MOON improved accuracy by {metrics['Accuracy']['Improvement (%)']:.2f}% compared to Base MOON.\n")
            else:
                f.write(f"BN Drift Control MOON decreased accuracy by {abs(metrics['Accuracy']['Improvement (%)']):.2f}% compared to Base MOON.\n")
        
        if "loss" in metrics:
            if metrics["Loss"]["Improvement (%)"] > 0:
                f.write(f"BN Drift Control MOON improved loss by {metrics['Loss']['Improvement (%)']:.2f}% compared to Base MOON.\n")
            else:
                f.write(f"BN Drift Control MOON increased loss by {abs(metrics['Loss']['Improvement (%)']):.2f}% compared to Base MOON.\n")
    
    print(f"Comparison results saved to {save_path}")
    print(f"CSV: {csv_path}")
    print(f"Excel: {excel_path}")
    print(f"Report: {report_path}")


def plot_comparison_results(
    base_history: History,
    bn_drift_history: History,
    save_plot_path: Path,
    experiment_name: str = "moon_comparison"
) -> None:
    """Plot comparison results between base MOON and BN drift control MOON.
    
    Parameters
    ----------
    base_history : History
        History object from base MOON experiment
    bn_drift_history : History
        History object from BN drift control MOON experiment
    save_plot_path : Path
        Path to save the comparison plots
    experiment_name : str, optional
        Name of the experiment, by default "moon_comparison"
    """
    import os
    from datetime import datetime
    
    # Create save directory if it doesn't exist
    os.makedirs(save_plot_path, exist_ok=True)
    
    # Create timestamp for unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Extract metrics
    base_rounds, base_values = zip(*base_history.metrics_centralized["accuracy"])
    bn_drift_rounds, bn_drift_values = zip(*bn_drift_history.metrics_centralized["accuracy"])
    
    # Plot accuracy comparison
    plt.figure(figsize=(10, 6))
    plt.plot(base_rounds, base_values, label="Base MOON")
    plt.plot(bn_drift_rounds, bn_drift_values, label="BN Drift Control MOON")
    plt.xlabel("#round")
    plt.ylabel("Test accuracy")
    plt.title(f"Accuracy Comparison: {experiment_name}")
    plt.legend()
    plt.grid(True)
    
    # Save plot
    plt.savefig(save_plot_path / f"{experiment_name}_accuracy_{timestamp}.png")
    plt.close()
    
    # Extract loss metrics if available
    if "loss" in base_history.metrics_centralized and "loss" in bn_drift_history.metrics_centralized:
        base_loss_rounds, base_loss_values = zip(*base_history.metrics_centralized["loss"])
        bn_drift_loss_rounds, bn_drift_loss_values = zip(*bn_drift_history.metrics_centralized["loss"])
        
        # Plot loss comparison
        plt.figure(figsize=(10, 6))
        plt.plot(base_loss_rounds, base_loss_values, label="Base MOON")
        plt.plot(bn_drift_loss_rounds, bn_drift_loss_values, label="BN Drift Control MOON")
        plt.xlabel("#round")
        plt.ylabel("Test loss")
        plt.title(f"Loss Comparison: {experiment_name}")
        plt.legend()
        plt.grid(True)
        
        # Save plot
        plt.savefig(save_plot_path / f"{experiment_name}_loss_{timestamp}.png")
        plt.close()
    
    print(f"Comparison plots saved to {save_plot_path}")
