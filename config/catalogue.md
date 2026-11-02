# Config value catalogue

| Config key | Possible values |
| --- | --- |
| `processor_config.skip_processing` | `true`, `false` |
| `processor_config.intersection_distributions` | `no_intersection`, `point_intersection`, `segment_intersection`, `polygon_intersection`, `polyhedron_intersection` (percentages should sum to 100) |
| `processor_config.transformations` | `[]`, `principal_axis`, `unitary_tetrahedron` |
| `model_config.task` | `IntersectionStatus`, `IntersectionVolume`, `IntersectionStatus_IntersectionVolume` |
| `model_config.architecture.use_model` | `mlp`, `tetrahedronpairnet` |
| `model_config.architecture.tetrahedronpairnet.vertices_aggregation_function` | `max`, `mean`, `sum`, `min`, `hadamard_prod` |
| `model_config.architecture.tetrahedronpairnet.tetrahedra_aggregation_function` | `max`, `mean`, `sum`, `min`, `hadamard_prod` |
| `model_config.common_parameters.activation_function` | `relu`, `tanh`, `leakyrelu`, `elu`, `gelu`, `swish`, `mish` |
| `trainer_config.skip_training`, `evaluator_config.skip_evaluation` | `true`, `false` |
| loss function | fixed by `task`: `IntersectionStatus` uses BCE, `IntersectionVolume` uses RMSLE, `IntersectionStatus_IntersectionVolume` uses the equal-weight sum of both |

