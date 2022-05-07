general_arguments = [
    'gpu_id', 'use_gpu',
    'seed',
    'reproducibility',
    'state',
    'data_path',
    'checkpoint_dir',
    'show_progress',
    'dataset_save_path',
    'config_file',
]

dataset_arguments = [
    'field_separator', 'seq_separator',
]

training_arguments = [
    'epochs', 'train_batch_size',
    'learner', 'learning_rate',
    'eval_step', 'stopping_step',
    'clip_grad_norm',
    'weight_decay',
    'loss_decimal_place',
]

evaluation_arguments = [
    'eval_args', 'repeatable',
    'metrics', 'topk', 'valid_metric', 'valid_metric_bigger',
    'eval_batch_size',
    'metric_decimal_place',
]
