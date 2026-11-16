"""Age-group-specific processing profiles for the Q-CARE ML pipeline.

Each profile defines:
- Feature normalization parameters (calibrated per age group).
- QAOA configuration (n_qubits, depth, target_k).
- VQC configuration (n_qubits, depth).
- Output risk thresholds.
- Validation rules.

IMPORTANT: These profiles are research defaults. They must be replaced by
values derived from actual labelled data before clinical use. The defaults
are structurally correct placeholders based on documented developmental
psychology literature (attention span, screen-time norms, motor activity).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# Age group constants (must match models_auth.py)
AGE_GROUP_A = "5_7"   # 5.0 ≤ age < 7.0
AGE_GROUP_B = "7_10"  # 7.0 ≤ age < 10.0
AGE_GROUP_C = "10_12" # 10.0 ≤ age ≤ 12.0

VALID_AGE_GROUPS = {AGE_GROUP_A, AGE_GROUP_B, AGE_GROUP_C}

AGE_GROUP_LABELS = {
    AGE_GROUP_A: "5–7 Years",
    AGE_GROUP_B: "7–10 Years",
    AGE_GROUP_C: "10–12 Years",
}


@dataclass
class AgeGroupProfile:
    """Processing configuration for one age group.

    Fields are research-calibrated per group. Defaults are structured placeholders;
    update from trained model metadata before production use.
    """
    age_group: str
    label: str

    # --- Feature normalization ---
    # z-score parameters: (mean, std) per fused feature index; None → use dataset stats
    feature_mean: Optional[list[float]] = None
    feature_std: Optional[list[float]] = None

    # --- QAOA configuration ---
    qaoa_n_qubits: int = 4
    qaoa_depth: int = 2
    qaoa_target_k: int = 2      # number of features to select
    qaoa_max_iter: int = 100

    # --- VQC configuration ---
    vqc_n_qubits: int = 4
    vqc_depth: int = 3
    vqc_num_classes: int = 3    # Low / Moderate / High risk

    # --- Risk thresholds (applied to softmax output) ---
    # Indices correspond to class_names: ["Low", "Moderate", "High"]
    # Threshold to report "High risk" flag
    high_risk_threshold: float = 0.60
    # Minimum confidence to report any prediction
    min_confidence: float = 0.40

    # --- Validation rules ---
    # Minimum number of tracked crops required for a valid assessment
    min_tracked_frames: int = 5
    # Maximum acceptable proportion of lost frames
    max_lost_frame_ratio: float = 0.30

    # --- Sequence processing ---
    sequence_length: int = 30
    frame_height: int = 112
    frame_width: int = 112

    # --- Age-specific notes (for audit / display) ---
    research_notes: str = ""


# ---------------------------------------------------------------------------
# Default profiles — research-informed structural defaults
# ---------------------------------------------------------------------------

PROFILES: dict[str, AgeGroupProfile] = {
    AGE_GROUP_A: AgeGroupProfile(
        age_group=AGE_GROUP_A,
        label="5–7 Years",
        # Young children: shorter attention span, more movement, simpler behaviour patterns.
        # Shorter sequences tolerated; more lenient on lost frames (child is active).
        min_tracked_frames=4,
        max_lost_frame_ratio=0.40,
        sequence_length=20,
        qaoa_n_qubits=4,
        qaoa_depth=1,
        qaoa_target_k=2,
        vqc_n_qubits=4,
        vqc_depth=2,
        high_risk_threshold=0.55,
        min_confidence=0.35,
        research_notes=(
            "Profile A (5-7 years): designed for younger children with shorter "
            "sustained attention. Screen-time norms: WHO recommends <1h/day for 5-year-olds. "
            "Thresholds are research defaults; validate with labelled data."
        ),
    ),
    AGE_GROUP_B: AgeGroupProfile(
        age_group=AGE_GROUP_B,
        label="7–10 Years",
        # Middle childhood: more stable attention, clearer behavioural patterns.
        min_tracked_frames=5,
        max_lost_frame_ratio=0.30,
        sequence_length=30,
        qaoa_n_qubits=4,
        qaoa_depth=2,
        qaoa_target_k=2,
        vqc_n_qubits=4,
        vqc_depth=3,
        high_risk_threshold=0.60,
        min_confidence=0.40,
        research_notes=(
            "Profile B (7-10 years): middle childhood. Standard configuration. "
            "Screen-time association research most studied in this range. "
            "Thresholds are research defaults; validate with labelled data."
        ),
    ),
    AGE_GROUP_C: AgeGroupProfile(
        age_group=AGE_GROUP_C,
        label="10–12 Years",
        # Older children: more voluntary control, complex social media exposure.
        # More stringent tracking requirements.
        min_tracked_frames=6,
        max_lost_frame_ratio=0.25,
        sequence_length=30,
        qaoa_n_qubits=4,
        qaoa_depth=2,
        qaoa_target_k=3,   # slightly more features for more complex behaviour
        vqc_n_qubits=4,
        vqc_depth=3,
        high_risk_threshold=0.65,
        min_confidence=0.45,
        research_notes=(
            "Profile C (10-12 years): preadolescent. More complex behavioural indicators. "
            "Social media and gaming exposure more prevalent. "
            "Thresholds are research defaults; validate with labelled data."
        ),
    ),
}


def get_profile(age_group: str) -> AgeGroupProfile:
    """Return the processing profile for the given age group. Raises ValueError if unknown."""
    if age_group not in PROFILES:
        raise ValueError(
            f"Unknown age_group '{age_group}'. Valid values: {list(PROFILES.keys())}"
        )
    return PROFILES[age_group]


def update_profile_from_model_config(age_group: str, config: dict) -> None:
    """Update a profile with trained-model normalization/threshold parameters.

    Called when a model is registered with per-age-group calibration data.
    """
    if age_group not in PROFILES:
        raise ValueError(f"Unknown age_group: {age_group}")
    profile = PROFILES[age_group]
    if "feature_mean" in config:
        profile.feature_mean = config["feature_mean"]
    if "feature_std" in config:
        profile.feature_std = config["feature_std"]
    if "high_risk_threshold" in config:
        profile.high_risk_threshold = float(config["high_risk_threshold"])
    if "min_confidence" in config:
        profile.min_confidence = float(config["min_confidence"])
    if "qaoa_target_k" in config:
        profile.qaoa_target_k = int(config["qaoa_target_k"])
