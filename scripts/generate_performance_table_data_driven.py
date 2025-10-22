#!/usr/bin/env python3
"""
Generate Data-Driven Performance Comparison Table for README

Creates performance comparison tables using actual experimental data from runs/ directory.
All metrics are computed from real experimental results with full traceability.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import glob


def load_experimental_data(runs_dir: Path) -> Dict:
    """Load and analyze experimental data from runs directory."""
    data = {
        "aggregation_methods": {},
        "heterogeneity_impact": {},
        "attack_resilience": {},
        "privacy_utility": {},
        "personalization_benefits": {},
        "data_sources": []
    }
    
    # Find all experimental runs
    run_dirs = [d for d in runs_dir.glob("comp_*") if d.is_dir()]
    print(f"Found {len(run_dirs)} experimental runs")
    
    for run_dir in run_dirs:
        run_name = run_dir.name
        config_file = run_dir / "config.json"
        metrics_file = run_dir / "metrics.csv"
        
        if not config_file.exists() or not metrics_file.exists():
            continue
            
        # Load configuration
        try:
            with open(config_file) as f:
                config = json.load(f)
        except:
            continue
            
        # Load metrics
        try:
            metrics_df = pd.read_csv(metrics_file)
        except:
            continue
            
        # Load client metrics
        client_metrics = []
        for client_file in run_dir.glob("client_*_metrics.csv"):
            try:
                client_df = pd.read_csv(client_file)
                client_metrics.append(client_df)
            except:
                continue
                
        if not client_metrics:
            continue
            
        # Combine client metrics
        all_client_metrics = pd.concat(client_metrics, ignore_index=True)
        
        # Extract key parameters
        agg_method = config.get("aggregation", "unknown")
        alpha = config.get("alpha", 1.0)
        byzantine_f = config.get("adversary_fraction", 0.0)
        dp_epsilon = config.get("dp_noise_multiplier", None) if config.get("dp_enabled", False) else None
        personalization_epochs = config.get("personalization_epochs", 0)
        fedprox_mu = config.get("fedprox_mu", 0.0)
        
        # Compute performance metrics from actual data
        final_round = all_client_metrics["round"].max()
        final_metrics = all_client_metrics[all_client_metrics["round"] == final_round]
        
        if len(final_metrics) == 0:
            continue
            
        # Use accuracy as primary metric (F1 not available in current data)
        mean_accuracy = final_metrics["acc_after"].mean()
        std_accuracy = final_metrics["acc_after"].std()
        mean_loss = final_metrics["loss_after"].mean()
        
        # Store data with full traceability
        run_data = {
            "run_name": run_name,
            "config": config,
            "final_accuracy": mean_accuracy,
            "accuracy_std": std_accuracy,
            "final_loss": mean_loss,
            "n_clients": len(final_metrics),
            "final_round": final_round,
            "data_source": f"runs/{run_name}/"
        }
        
        # Categorize by experimental dimension
        if byzantine_f == 0 and dp_epsilon is None and personalization_epochs == 0:
            # Aggregation comparison
            if agg_method not in data["aggregation_methods"]:
                data["aggregation_methods"][agg_method] = []
            data["aggregation_methods"][agg_method].append(run_data)
            
        if byzantine_f > 0:
            # Attack resilience
            if agg_method not in data["attack_resilience"]:
                data["attack_resilience"][agg_method] = {}
            if byzantine_f not in data["attack_resilience"][agg_method]:
                data["attack_resilience"][agg_method][byzantine_f] = []
            data["attack_resilience"][agg_method][byzantine_f].append(run_data)
            
        if dp_epsilon is not None:
            # Privacy-utility tradeoff
            if agg_method not in data["privacy_utility"]:
                data["privacy_utility"][agg_method] = {}
            if dp_epsilon not in data["privacy_utility"][agg_method]:
                data["privacy_utility"][agg_method][dp_epsilon] = []
            data["privacy_utility"][agg_method][dp_epsilon].append(run_data)
            
        if personalization_epochs > 0:
            # Personalization benefits
            if agg_method not in data["personalization_benefits"]:
                data["personalization_benefits"][agg_method] = {}
            if personalization_epochs not in data["personalization_benefits"][agg_method]:
                data["personalization_benefits"][agg_method][personalization_epochs] = []
            data["personalization_benefits"][agg_method][personalization_epochs].append(run_data)
            
        # Heterogeneity impact (FedAvg vs FedProx)
        if fedprox_mu > 0 or agg_method == "fedavg":
            if alpha not in data["heterogeneity_impact"]:
                data["heterogeneity_impact"][alpha] = {}
            data["heterogeneity_impact"][alpha][agg_method] = run_data
            
        data["data_sources"].append(run_data["data_source"])
    
    return data


def create_aggregation_comparison_table(data: Dict) -> str:
    """Create aggregation methods comparison table using real data."""
    if not data["aggregation_methods"]:
        return "### Aggregation Methods Performance\n\n*No aggregation comparison data available in current experimental runs*\n"
    
    table = "### Aggregation Methods Performance\n\n"
    table += "| Method | Accuracy | Loss | Clients | Data Source |\n"
    table += "|--------|----------|------|---------|-------------|\n"
    
    for method, runs in data["aggregation_methods"].items():
        if not runs:
            continue
            
        # Compute statistics across all runs for this method
        accuracies = [r["final_accuracy"] for r in runs]
        losses = [r["final_loss"] for r in runs]
        
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        mean_loss = np.mean(losses)
        n_clients = runs[0]["n_clients"] if runs else 0
        
        # Get data source
        data_source = runs[0]["data_source"] if runs else "N/A"
        
        table += f"| **{method.upper()}** | {mean_acc:.3f} ± {std_acc:.3f} | {mean_loss:.3f} | {n_clients} | [{data_source}]({data_source}) |\n"
    
    table += "\n*Performance metrics computed from final round accuracy and loss values*\n"
    table += "*Note: F1 scores not available in current experimental data - using accuracy as primary metric*\n"
    
    return table


def create_heterogeneity_impact_table(data: Dict) -> str:
    """Create heterogeneity impact table using real data."""
    if not data["heterogeneity_impact"]:
        return "### Data Heterogeneity Impact\n\n*No heterogeneity comparison data available in current experimental runs*\n"
    
    table = "### Data Heterogeneity Impact\n\n"
    table += "| α Value | FedAvg Accuracy | FedProx Accuracy | Improvement | Data Source |\n"
    table += "|---------|-----------------|------------------|-------------|-------------|\n"
    
    for alpha in sorted(data["heterogeneity_impact"].keys()):
        alpha_data = data["heterogeneity_impact"][alpha]
        
        fedavg_acc = None
        fedprox_acc = None
        fedavg_source = None
        fedprox_source = None
        
        if "fedavg" in alpha_data:
            fedavg_acc = alpha_data["fedavg"]["final_accuracy"]
            fedavg_source = alpha_data["fedavg"]["data_source"]
            
        if "fedprox" in alpha_data:
            fedprox_acc = alpha_data["fedprox"]["final_accuracy"]
            fedprox_source = alpha_data["fedprox"]["data_source"]
            
        if fedavg_acc is not None and fedprox_acc is not None:
            improvement = ((fedprox_acc - fedavg_acc) / fedavg_acc) * 100
            table += f"| {alpha} | {fedavg_acc:.3f} | {fedprox_acc:.3f} | {improvement:+.1f}% | [FedAvg]({fedavg_source}) / [FedProx]({fedprox_source}) |\n"
        elif fedavg_acc is not None:
            table += f"| {alpha} | {fedavg_acc:.3f} | N/A | - | [FedAvg]({fedavg_source}) |\n"
        elif fedprox_acc is not None:
            table += f"| {alpha} | N/A | {fedprox_acc:.3f} | - | [FedProx]({fedprox_source}) |\n"
    
    table += "\n*Heterogeneity levels: α=1.0 (IID), α=0.5 (mild non-IID), α=0.1 (severe non-IID)*\n"
    table += "*Note: Using accuracy as primary metric - F1 scores not available in current data*\n"
    
    return table


def create_attack_resilience_table(data: Dict) -> str:
    """Create attack resilience table using real data."""
    if not data["attack_resilience"]:
        return "### Attack Resilience Performance\n\n*No attack resilience data available in current experimental runs*\n"
    
    table = "### Attack Resilience Performance\n\n"
    table += "| Method | Clean Data | 10% Byzantine | 30% Byzantine | Data Sources |\n"
    table += "|--------|------------|---------------|---------------|-------------|\n"
    
    for method, byzantine_data in data["attack_resilience"].items():
        clean_acc = None
        byz_10_acc = None
        byz_30_acc = None
        sources = []
        
        for byz_fraction, runs in byzantine_data.items():
            if not runs:
                continue
                
            mean_acc = np.mean([r["final_accuracy"] for r in runs])
            source = runs[0]["data_source"]
            sources.append(f"[{byz_fraction*100:.0f}%]({source})")
            
            if byz_fraction == 0.0:
                clean_acc = mean_acc
            elif byz_fraction == 0.1:
                byz_10_acc = mean_acc
            elif byz_fraction == 0.3:
                byz_30_acc = mean_acc
        
        clean_str = f"{clean_acc:.3f}" if clean_acc is not None else "N/A"
        byz_10_str = f"{byz_10_acc:.3f}" if byz_10_acc is not None else "N/A"
        byz_30_str = f"{byz_30_acc:.3f}" if byz_30_acc is not None else "N/A"
        sources_str = " / ".join(sources) if sources else "N/A"
        
        table += f"| **{method.upper()}** | {clean_str} | {byz_10_str} | {byz_30_str} | {sources_str} |\n"
    
    table += "\n*Performance under Byzantine attacks - accuracy values from final round*\n"
    table += "*Note: Using accuracy as primary metric - F1 scores not available in current data*\n"
    
    return table


def create_personalization_benefits_table(data: Dict) -> str:
    """Create personalization benefits table using real data."""
    # Load existing personalization analysis if available
    pers_file = Path("analysis/personalization/personalization_summary.json")
    if pers_file.exists():
        with open(pers_file) as f:
            pers_data = json.load(f)
    else:
        pers_data = None
    
    table = "### Personalization Benefits\n\n"
    
    if pers_data:
        table += "| Scenario | Mean Gain | Clients with Gains | Data Source |\n"
        table += "|----------|-----------|-------------------|-------------|\n"
        table += f"| **Overall** | {pers_data['overall']['mean_gain']:.1%} | {pers_data['overall']['pct_positive_gains']:.0f}% | [analysis/personalization/](analysis/personalization/) |\n"
        table += f"| **CIC-IDS2017** | {pers_data['by_dataset']['cic']['mean_gain']:.1%} | - | [analysis/personalization/](analysis/personalization/) |\n"
        table += f"| **UNSW-NB15** | {pers_data['by_dataset']['unsw']['mean_gain']:.1%} | - | [analysis/personalization/](analysis/personalization/) |\n"
        table += f"| **Severe Non-IID (α=0.1)** | {pers_data['by_alpha']['0.1']['mean_gain']:.1%} | - | [analysis/personalization/](analysis/personalization/) |\n"
        table += f"| **5 Personalization Epochs** | {pers_data['by_epochs']['5']['mean_gain']:.1%} | - | [analysis/personalization/](analysis/personalization/) |\n"
        table += "\n*Personalization gains measured as F1 score improvement over global model*\n"
    else:
        table += "*Personalization analysis data not available in current experimental runs*\n"
    
    return table


def create_privacy_utility_table(data: Dict) -> str:
    """Create privacy-utility tradeoff table using real data."""
    if not data["privacy_utility"]:
        return "### Privacy-Utility Tradeoff\n\n*No privacy-utility data available in current experimental runs*\n"
    
    table = "### Privacy-Utility Tradeoff\n\n"
    table += "| Method | ε (Epsilon) | Accuracy | Loss | Data Source |\n"
    table += "|--------|-------------|----------|------|-------------|\n"
    
    for method, epsilon_data in data["privacy_utility"].items():
        for epsilon, runs in epsilon_data.items():
            if not runs:
                continue
                
            mean_acc = np.mean([r["final_accuracy"] for r in runs])
            mean_loss = np.mean([r["final_loss"] for r in runs])
            source = runs[0]["data_source"]
            
            table += f"| **{method.upper()}** | {epsilon} | {mean_acc:.3f} | {mean_loss:.3f} | [{source}]({source}) |\n"
    
    table += "\n*Differential privacy impact on model performance*\n"
    table += "*Note: Using accuracy as primary metric - F1 scores not available in current data*\n"
    
    return table


def create_methodology_section(data: Dict) -> str:
    """Create methodology section explaining data sources and computation."""
    methodology = "## Methodology & Data Sources\n\n"
    methodology += "### Data Collection\n\n"
    methodology += f"- **Total Experimental Runs**: {len(data['data_sources'])}\n"
    methodology += f"- **Data Directory**: [`runs/`](runs/)\n"
    methodology += f"- **Primary Metric**: Accuracy (F1 scores not available in current experimental data)\n"
    methodology += f"- **Computation Method**: Mean accuracy across all clients in final training round\n\n"
    
    methodology += "### Available Experimental Dimensions\n\n"
    
    if data["aggregation_methods"]:
        methods = list(data["aggregation_methods"].keys())
        methodology += f"- **Aggregation Methods**: {', '.join(methods)}\n"
    
    if data["heterogeneity_impact"]:
        alphas = sorted(data["heterogeneity_impact"].keys())
        methodology += f"- **Heterogeneity Levels**: α = {', '.join(map(str, alphas))}\n"
    
    if data["attack_resilience"]:
        byzantine_levels = set()
        for method_data in data["attack_resilience"].values():
            byzantine_levels.update(method_data.keys())
        byzantine_levels = sorted(byzantine_levels)
        methodology += f"- **Byzantine Attack Levels**: {', '.join([f'{b*100:.0f}%' for b in byzantine_levels])}\n"
    
    if data["privacy_utility"]:
        epsilon_levels = set()
        for method_data in data["privacy_utility"].values():
            epsilon_levels.update(method_data.keys())
        epsilon_levels = sorted(epsilon_levels)
        methodology += f"- **Privacy Levels**: ε = {', '.join(map(str, epsilon_levels))}\n"
    
    methodology += "\n### Data Traceability\n\n"
    methodology += "All performance claims are linked to specific experimental runs:\n"
    for i, source in enumerate(data["data_sources"][:10], 1):  # Show first 10
        methodology += f"{i}. [{source}]({source})\n"
    
    if len(data["data_sources"]) > 10:
        methodology += f"... and {len(data['data_sources']) - 10} more runs\n"
    
    methodology += "\n### Limitations\n\n"
    methodology += "- **F1 Scores**: Not available in current experimental data - using accuracy as primary metric\n"
    methodology += "- **Statistical Significance**: No confidence intervals computed (future enhancement)\n"
    methodology += "- **Cross-Validation**: Single train/test split per experiment\n"
    
    return methodology


def main():
    """Generate data-driven performance comparison tables."""
    runs_dir = Path("runs")
    
    if not runs_dir.exists():
        print("Error: runs/ directory not found")
        return
    
    print("Loading experimental data...")
    data = load_experimental_data(runs_dir)
    
    print(f"Loaded data from {len(data['data_sources'])} experimental runs")
    
    # Generate all tables
    tables = {
        "aggregation": create_aggregation_comparison_table(data),
        "heterogeneity": create_heterogeneity_impact_table(data),
        "attack_resilience": create_attack_resilience_table(data),
        "personalization": create_personalization_benefits_table(data),
        "privacy_utility": create_privacy_utility_table(data),
        "methodology": create_methodology_section(data)
    }
    
    # Save tables to file
    output_file = Path("PERFORMANCE_COMPARISON_TABLES_DATA_DRIVEN.md")
    with open(output_file, "w") as f:
        f.write("# Data-Driven Performance Comparison Tables\n\n")
        f.write("Comprehensive performance analysis using actual experimental data with full traceability.\n\n")
        
        for name, table in tables.items():
            f.write(f"## {name.replace('_', ' ').title()}\n")
            f.write(table)
            f.write("\n")
    
    print(f"Data-driven performance comparison tables saved to {output_file}")
    
    # Print summary
    print("\nGenerated tables:")
    for name in tables.keys():
        print(f"  - {name.replace('_', ' ').title()}")


if __name__ == "__main__":
    main()