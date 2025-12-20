# Cluster Runs Analysis

Source: `cluster-experiments/cluster-runs`

## Coverage Summary

- Total runs: 3131
- Runs with metrics.csv: 3129
- Runs missing metrics.csv: 2
- Runs missing final macro_f1_test despite metrics.csv: 1459

### Runs by aggregation

| Aggregation | Count |
| --- | --- |
| bulyan | 356 |
| fedavg | 1416 |
| fedprox | 511 |
| krum | 623 |
| median | 225 |

## Duplicate Configurations

- Duplicate config groups: 10
- Total runs in duplicate groups: 20

### Duplicate groups (first 20)

| Group | Count | Example run dir |
| --- | --- | --- |
| 1 | 2 | comp_fedavg_alpha0.5_adv0_dp1_pers0_mu0.0_seed46 |
| 2 | 2 | comp_fedavg_alpha0.5_adv0_dp1_dpnoise0.1_pers0_mu0.0_seed46_datasetcic |
| 3 | 2 | comp_fedavg_alpha0.5_adv0_dp1_dpnoise0.1_pers0_mu0.0_seed43_datasetcic |
| 4 | 2 | comp_fedavg_alpha0.5_adv0_dp1_pers0_mu0.0_seed45_datasetcic |
| 5 | 2 | comp_fedavg_alpha0.5_adv0_dp1_pers0_mu0.0_seed42_datasetcic |
| 6 | 2 | comp_fedavg_alpha0.5_adv0_dp1_dpnoise0.1_pers0_mu0.0_seed43 |
| 7 | 2 | comp_fedavg_alpha0.5_adv0_dp1_dpnoise0.1_pers0_mu0.0_seed44 |
| 8 | 2 | comp_fedavg_alpha0.5_adv0_dp1_dpnoise0.1_pers0_mu0.0_seed45 |
| 9 | 2 | comp_fedavg_alpha0.5_adv0_dp1_dpnoise0.1_pers0_mu0.0_seed42 |
| 10 | 2 | comp_fedavg_alpha0.5_adv0_dp1_pers0_mu0.0_seed44_datasetcic |

## Performance Summary (Final global_macro_f1_test)

### Mean and std by aggregation, alpha, adversary_fraction

| Aggregation | Alpha | Adversary fraction | N | Mean macro_f1_test | Std |
| --- | --- | --- | --- | --- | --- |
| bulyan | 0.02 | 0.0 | 20 | 0.398796 | 0.086988 |
| bulyan | 0.02 | 0.1 | 20 | 0.378651 | 0.113809 |
| bulyan | 0.05 | 0.0 | 20 | 0.511711 | 0.083885 |
| bulyan | 0.05 | 0.1 | 20 | 0.488008 | 0.091321 |
| bulyan | 0.1 | 0.0 | 20 | 0.594838 | 0.037995 |
| bulyan | 0.1 | 0.1 | 20 | 0.518845 | 0.113847 |
| bulyan | 0.2 | 0.0 | 20 | 0.630850 | 0.030149 |
| bulyan | 0.2 | 0.1 | 20 | 0.549770 | 0.108066 |
| bulyan | 0.5 | 0.0 | 20 | 0.666229 | 0.016436 |
| bulyan | 0.5 | 0.1 | 20 | 0.599180 | 0.064774 |
| bulyan | 0.5 | 0.3 | 0 |  |  |
| bulyan | 1.0 | 0.0 | 20 | 0.683805 | 0.013757 |
| bulyan | 1.0 | 0.1 | 20 | 0.628381 | 0.028245 |
| bulyan | inf | 0.0 | 20 | 0.703302 | 0.024704 |
| bulyan | inf | 0.1 | 4 | 0.524574 | 0.015723 |
| fedavg | 0.02 | 0.0 | 18 | 0.385394 | 0.068434 |
| fedavg | 0.02 | 0.1 | 20 | 0.207274 | 0.092794 |
| fedavg | 0.02 | 0.2 | 20 | 0.165686 | 0.084810 |
| fedavg | 0.02 | 0.3 | 20 | 0.109579 | 0.105839 |
| fedavg | 0.05 | 0.0 | 18 | 0.472988 | 0.075949 |
| fedavg | 0.05 | 0.1 | 20 | 0.247110 | 0.118791 |
| fedavg | 0.05 | 0.2 | 20 | 0.171082 | 0.109251 |
| fedavg | 0.05 | 0.3 | 20 | 0.111249 | 0.111793 |
| fedavg | 0.1 | 0.0 | 15 | 0.591866 | 0.041307 |
| fedavg | 0.1 | 0.1 | 20 | 0.264463 | 0.177596 |
| fedavg | 0.1 | 0.2 | 20 | 0.194348 | 0.136583 |
| fedavg | 0.1 | 0.3 | 20 | 0.095891 | 0.108950 |
| fedavg | 0.2 | 0.0 | 15 | 0.644464 | 0.020165 |
| fedavg | 0.2 | 0.1 | 20 | 0.283065 | 0.204646 |
| fedavg | 0.2 | 0.2 | 20 | 0.192655 | 0.183170 |
| fedavg | 0.2 | 0.3 | 20 | 0.105084 | 0.139511 |
| fedavg | 0.5 | 0.0 | 15 | 0.663579 | 0.016268 |
| fedavg | 0.5 | 0.1 | 20 | 0.307301 | 0.212051 |
| fedavg | 0.5 | 0.2 | 20 | 0.163500 | 0.174856 |
| fedavg | 0.5 | 0.3 | 20 | 0.097541 | 0.146521 |
| fedavg | 1.0 | 0.0 | 13 | 0.683844 | 0.012551 |
| fedavg | 1.0 | 0.1 | 20 | 0.296958 | 0.177492 |
| fedavg | 1.0 | 0.2 | 20 | 0.147502 | 0.110849 |
| fedavg | 1.0 | 0.3 | 20 | 0.061279 | 0.050843 |
| fedavg | inf | 0.0 | 15 | 0.712985 | 0.008653 |
| fedavg | inf | 0.0 | 0 |  |  |
| fedavg | inf | 0.1 | 20 | 0.106789 | 0.015602 |
| fedavg | inf | 0.2 | 20 | 0.054567 | 0.008461 |
| fedavg | inf | 0.3 | 20 | 0.036683 | 0.004900 |
| fedprox | 0.02 | 0.0 | 33 | 0.457300 | 0.091580 |
| fedprox | 0.05 | 0.0 | 11 | 0.587102 | 0.036903 |
| fedprox | 0.1 | 0.0 | 11 | 0.655449 | 0.086634 |
| fedprox | 0.2 | 0.0 | 11 | 0.665013 | 0.024893 |
| fedprox | 0.5 | 0.0 | 11 | 0.686561 | 0.010744 |
| fedprox | 1.0 | 0.0 | 11 | 0.683430 | 0.010541 |
| fedprox | inf | 0.0 | 11 | 0.722149 | 0.007690 |
| fedprox | inf | 0.0 | 0 |  |  |
| krum | 0.02 | 0.0 | 20 | 0.281526 | 0.071166 |
| krum | 0.02 | 0.1 | 20 | 0.258260 | 0.079913 |
| krum | 0.02 | 0.2 | 20 | 0.246929 | 0.095326 |
| krum | 0.02 | 0.3 | 20 | 0.197376 | 0.104077 |
| krum | 0.05 | 0.0 | 20 | 0.297442 | 0.051208 |
| krum | 0.05 | 0.1 | 20 | 0.270482 | 0.059305 |
| krum | 0.05 | 0.2 | 20 | 0.239313 | 0.069859 |
| krum | 0.05 | 0.3 | 20 | 0.200680 | 0.085771 |
| krum | 0.1 | 0.0 | 20 | 0.406903 | 0.096207 |
| krum | 0.1 | 0.1 | 20 | 0.373017 | 0.118051 |
| krum | 0.1 | 0.2 | 20 | 0.313843 | 0.118987 |
| krum | 0.1 | 0.3 | 20 | 0.237259 | 0.084138 |
| krum | 0.2 | 0.0 | 20 | 0.546627 | 0.042507 |
| krum | 0.2 | 0.1 | 20 | 0.473980 | 0.091969 |
| krum | 0.2 | 0.2 | 20 | 0.435978 | 0.097171 |
| krum | 0.2 | 0.3 | 20 | 0.366631 | 0.109339 |
| krum | 0.5 | 0.0 | 20 | 0.624136 | 0.029748 |
| krum | 0.5 | 0.1 | 20 | 0.563461 | 0.057236 |
| krum | 0.5 | 0.2 | 20 | 0.494289 | 0.077415 |
| krum | 0.5 | 0.3 | 20 | 0.417571 | 0.096856 |
| krum | 1.0 | 0.0 | 20 | 0.646143 | 0.021982 |
| krum | 1.0 | 0.1 | 20 | 0.587159 | 0.031627 |
| krum | 1.0 | 0.2 | 20 | 0.536621 | 0.031729 |
| krum | 1.0 | 0.3 | 20 | 0.471217 | 0.043218 |
| krum | inf | 0.0 | 20 | 0.693702 | 0.021530 |
| krum | inf | 0.1 | 20 | 0.628144 | 0.016110 |
| krum | inf | 0.2 | 20 | 0.556509 | 0.018311 |
| krum | inf | 0.3 | 20 | 0.489892 | 0.014115 |
| median | 0.02 | 0.2 | 20 | 0.334149 | 0.117007 |
| median | 0.02 | 0.3 | 20 | 0.222483 | 0.116195 |
| median | 0.05 | 0.2 | 5 | 0.308904 | 0.033121 |
| median | 0.5 | 0.0 | 19 | 0.670978 | 0.016660 |
| median | 0.5 | 0.1 | 20 | 0.593212 | 0.061552 |
| median | 0.5 | 0.3 | 0 |  |  |
| median | 1.0 | 0.0 | 20 | 0.685024 | 0.011858 |
| median | 1.0 | 0.1 | 20 | 0.623686 | 0.027105 |
| median | inf | 0.0 | 20 | 0.697843 | 0.021966 |
| median | inf | 0.1 | 20 | 0.628599 | 0.018650 |

## Notes

- Duplicate detection uses full config.json content; if you want duplicates collapsed by a subset (e.g., ignoring data_path), rerun with that rule.
- Final macro_f1_test uses the last non-empty value in metrics.csv; if you want a specific round selection rule, we can tighten this.
