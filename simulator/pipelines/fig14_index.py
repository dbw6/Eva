import pandas as pd

from simulator.aqlm_analysis import (
    compute_average_index_counts,
    compute_avg_unique_per_tile,
    extract_weight_indices,
    load_quantized_model,
)
from simulator.methods import MethodRegistry
from simulator.models import ModelRegistry
from simulator.specs import RunnerConfig, StudyArtifacts, StudySpec


class IndexAnalysisPipeline:
    def run(
        self,
        config: RunnerConfig,
        study: StudySpec,
        model_registry: ModelRegistry,
        method_registry: MethodRegistry,
    ) -> StudyArtifacts:
        del model_registry, method_registry
        output_dir = config.output_dir / study.output_subdir
        output_dir.mkdir(parents=True, exist_ok=True)

        model_name = str(study.extra.get("aqlm_model_name", "ISTA-DASLab/Llama-2-7b-AQLM-2Bit-2x8-hf"))
        layer_number = int(study.extra.get("layer_number", 15))
        weight_type = str(study.extra.get("weight_type", "q_proj"))
        tile_sizes = [int(value) for value in study.extra.get("tile_sizes", [128, 256, 512, 1024, 2048, 4096])]

        model = load_quantized_model(model_name)
        indices_full = extract_weight_indices(model, layer_number, weight_type)
        avg_counts = compute_average_index_counts(indices_full)
        tile_results = compute_avg_unique_per_tile(indices_full, tile_sizes)

        histogram = pd.DataFrame(
            [{"index": idx, "avg_count": float(value)} for idx, value in enumerate(avg_counts)]
        )
        unique_tiles = pd.DataFrame(
            [{"tile_size": tile_size, "avg_unique_indices": avg_unique} for tile_size, avg_unique in tile_results]
        )
        histogram["avg_count"] = histogram["avg_count"].round(6)
        unique_tiles["avg_unique_indices"] = unique_tiles["avg_unique_indices"].round(4)

        histogram_csv = output_dir / "fig14_index_count_histogram_avg.csv"
        unique_csv = output_dir / "fig14_unique_indices_per_tile_avg.csv"
        histogram.to_csv(histogram_csv, index=False)
        unique_tiles.to_csv(unique_csv, index=False)

        return StudyArtifacts(
            output_dir=output_dir,
            cycles_csv=histogram_csv,
            energy_csv=unique_csv,
            power_csv=unique_csv,
            verification_json=None,
            reports={"histogram_csv": histogram_csv, "unique_tiles_csv": unique_csv},
        )
