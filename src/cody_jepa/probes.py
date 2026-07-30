"""Compatibility imports for the reorganized evaluation package.

New code should import feature utilities from :mod:`cody_jepa.evaluation.features`
and probe protocols from :mod:`cody_jepa.evaluation.probes`.
"""

from .evaluation.features import (
    FEATURE_FORMULA,
    FEATURE_SOURCE,
    INTEGER_METADATA_COLUMNS,
    METADATA_COLUMNS,
    NUMERIC_METADATA_COLUMNS,
    _atomic_path,
    _batch_values,
    _feature_columns,
    _sidecar_path,
    _write_json_atomic,
    build_frozen_target_encoder,
    export_frozen_features,
    read_feature_table,
    validate_feature_table,
    write_feature_table,
)
from .evaluation.probes.common import (
    classification_metrics as _classification_metrics,
    linear_predictions as _linear_predictions,
    majority_baseline as _majority_baseline,
)
from .evaluation.probes.gait import evaluate_gait_system
from .evaluation.probes.identity import (
    closed_set_masks as _closed_set_masks,
    evaluate_identity_closed_set,
    evaluate_identity_heldout_retrieval,
)
from .evaluation.probes.pipeline import (
    PROBE_SUMMARY_COLUMNS,
    evaluate_all_probes,
    write_probe_results,
)

__all__ = [
    "FEATURE_FORMULA",
    "FEATURE_SOURCE",
    "METADATA_COLUMNS",
    "PROBE_SUMMARY_COLUMNS",
    "build_frozen_target_encoder",
    "evaluate_all_probes",
    "evaluate_gait_system",
    "evaluate_identity_closed_set",
    "evaluate_identity_heldout_retrieval",
    "export_frozen_features",
    "read_feature_table",
    "validate_feature_table",
    "write_feature_table",
    "write_probe_results",
]
