# Config value catalogue

| Config key | Possible values |
| --- | --- |
| `processor_config.skip_processing` | `true`, `false` |
| `processor_config.sampling_strategy` | `log_uniform_volume`, `hybrid_volume` (anything else falls back to the regular/uniform sampler) |
| `processor_config.intersection_distributions` | `no_intersection`, `point_intersection`, `segment_intersection`, `polygon_intersection`, `polyhedron_intersection` (percentages should sum to 100) |
| `processor_config.transformations` | `[]`, `principal_axis`, `unitary_tetrahedron` |
| `processor_config.augmentations.*` | percentages in `0..100`, e.g. `point_wise_permutation_augmentation_pct`, `tetrahedron_wise_permutation_augmentation_pct` |
| `model_config.task` | `IntersectionStatus`, `IntersectionVolume`, `IntersectionStatus_IntersectionVolume` |
| `model_config.architecture.use_model` | `mlp`, `tetrahedronpairnet` |
| `model_config.architecture.tetrahedronpairnet.vertices_aggregation_function` | `max`, `mean`, `sum`, `min`, `hadamard_prod` |
| `model_config.architecture.tetrahedronpairnet.tetrahedra_aggregation_function` | `max`, `mean`, `sum`, `min`, `hadamard_prod` |
| `model_config.common_parameters.activation_function` | `relu`, `tanh`, `leakyrelu`, `elu`, `gelu`, `swish`, `mish` |
| `trainer_config.skip_training`, `trainer_config.fine_tune_on`, `evaluator_config.skip_evaluation` | `true`, `false` |
| `trainer_config.loss_function` | `IntersectionStatus`: `bce`; `IntersectionVolume`: `decade_weighted`, `rmsle`; `IntersectionStatus_IntersectionVolume`: `combined`, `volume_focused`, `volume_focused_strong_invariance`, `logcosh_aggressive`, `quantile_logcosh_aggressive`, `quantile_logcosh_strong_underpred`, `quantile_rmsle_aggressive` |
| `trainer_config.optimizer` | currently ignored by the code; training hardcodes `AdamW` |

