from pathlib import Path

from simulator.config import CONFIG_ROOT, load_yaml_directory
from simulator.specs import StudySpec


class StudyRegistry:
    def __init__(self, config_root: Path | None = None) -> None:
        study_root = (config_root or CONFIG_ROOT) / "studies"
        raw_studies = load_yaml_directory(study_root, "studies")
        self._studies = {name: self._build_spec(name, payload) for name, payload in raw_studies.items()}

    def get(self, study_name: str) -> StudySpec:
        if study_name not in self._studies:
            raise ValueError(f"Unknown study: {study_name}")
        return self._studies[study_name]

    def names(self) -> list[str]:
        return sorted(self._studies)

    def _build_spec(self, name: str, payload: dict) -> StudySpec:
        return StudySpec(
            name=name,
            description=payload["description"],
            phase=payload["phase"],
            ops_mode=payload["ops_mode"],
            models=tuple(payload.get("models", [])),
            methods=tuple(payload.get("methods", [])),
            sequence_lengths=tuple(payload.get("sequence_lengths", [])),
            batch_sizes=tuple(payload.get("batch_sizes", [1])),
            aggregate_sequence1=payload.get("aggregate_sequence1", False),
            output_subdir=payload.get("output_subdir", name),
            extra=payload.get("extra", {}),
        )
