import csv

from client_metrics import ClientMetricsLogger


def test_log_round_metrics_includes_attack_and_holdout(tmp_path) -> None:
    csv_path = tmp_path / "metrics.csv"
    logger = ClientMetricsLogger(str(csv_path), client_id=1, extended=True)

    logger.log_round_metrics(
        round_num=1,
        dataset_size=10,
        n_classes=2,
        loss_before=0.1,
        acc_before=0.9,
        loss_after=0.05,
        acc_after=0.95,
        macro_f1_before=0.8,
        macro_f1_after=0.85,
        macro_f1_argmax=0.85,
        benign_fpr_argmax=0.1,
        f1_per_class_after_json='{"0":0.8,"1":0.9}',
        precision_per_class_json='{"0":0.8,"1":0.9}',
        recall_per_class_json='{"0":0.8,"1":0.9}',
        confusion_matrix_counts_json="[[5,1],[0,4]]",
        confusion_matrix_normalized_json="[[0.83,0.17],[0.0,0.8]]",
        confusion_matrix_class_names_json='["benign","attack"]',
        fpr_after=0.1,
        pr_auc_after=0.75,
        threshold_tau=0.5,
        f1_bin_tau=0.8,
        benign_fpr_bin_tau=0.1,
        tau_bin=0.5,
        seed=42,
        weight_norm_before=1.0,
        weight_norm_after=1.1,
        weight_update_norm=0.1,
        grad_norm_l2=0.2,
        t_fit_ms=12.0,
        epochs_completed=1,
        lr=0.001,
        batch_size=4,
        dp_epsilon=1.5,
        dp_delta=1e-5,
        dp_sigma=0.1,
        dp_clip_norm=1.0,
        dp_sample_rate=0.5,
        dp_total_steps=10,
        dp_enabled_flag=True,
        attack_mode="sign_flip_topk",
        macro_f1_global_holdout=0.85,
        micro_f1_global_holdout=0.9,
        f1_per_class_holdout_json='{"0":0.8,"1":0.9}',
        confusion_matrix_counts_holdout_json="[[5,1],[0,4]]",
        confusion_matrix_normalized_holdout_json="[[0.83,0.17],[0.0,0.8]]",
        confusion_matrix_class_names_holdout_json='["benign","attack"]',
        secure_aggregation_flag=False,
    )

    with csv_path.open() as f:
        reader = csv.reader(f)
        header = next(reader)
        row = next(reader)
    values = dict(zip(header, row))

    assert values["attack_mode"] == "sign_flip_topk"
    assert values["macro_f1_global_holdout"] == "0.85"
    assert values["micro_f1_global_holdout"] == "0.9"
    assert values["f1_per_class_holdout"] == '{"0":0.8,"1":0.9}'
    assert values["confusion_matrix_counts_holdout"] == "[[5,1],[0,4]]"
    assert values["confusion_matrix_normalized_holdout"] == "[[0.83,0.17],[0.0,0.8]]"
    assert values["confusion_matrix_class_names_holdout"] == '["benign","attack"]'
