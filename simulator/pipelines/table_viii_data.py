import pandas as pd

from simulator.datasets import get_dataset_lengths
from simulator.methods import MethodRegistry
from simulator.models import ModelRegistry
from simulator.specs import RunnerConfig, StudyArtifacts, StudySpec


TABLE_VIII_COLUMNS = [
    "model",
    "dataset",
    "tokenizer",
    "sample_count",
    "avg_input_length",
    "avg_output_length",
]


class DatasetStatsPipeline:
    def run(
        self,
        config: RunnerConfig,
        study: StudySpec,
        model_registry: ModelRegistry,
        method_registry: MethodRegistry,
    ) -> StudyArtifacts:
        del model_registry, method_registry
        output_dir = config.output_dir / study.output_subdir
        tokenizer_map = dict(study.extra.get("tokenizers", {}))
        evaluations = list(study.extra.get("evaluations", []))
        seed = int(study.extra.get("seed", 42))

        rows: list[dict[str, object]] = []
        for evaluation in evaluations:
            model_name = str(evaluation["model"])
            dataset_name = str(evaluation["dataset"])
            tokenizer_name = str(tokenizer_map[model_name])
            sample_count = int(evaluation["sample_count"])
            lengths = get_dataset_lengths(dataset_name, tokenizer_name, sample_count, seed=seed)
            avg_input = sum(input_len for input_len, _ in lengths) / len(lengths)
            avg_output = sum(output_len for _, output_len in lengths) / len(lengths)
            rows.append(
                {
                    "model": model_name,
                    "dataset": dataset_name,
                    "tokenizer": tokenizer_name,
                    "sample_count": sample_count,
                    "avg_input_length": avg_input,
                    "avg_output_length": avg_output,
                }
            )

        dataframe = pd.DataFrame(rows)
        table_csv = output_dir / "table_viii.csv"
        table_csv.parent.mkdir(parents=True, exist_ok=True)
        dataframe[TABLE_VIII_COLUMNS].to_csv(table_csv, index=False)

        return StudyArtifacts(
            output_dir=output_dir,
            cycles_csv=table_csv,
            energy_csv=table_csv,
            power_csv=table_csv,
            verification_json=None,
            reports={"table_viii_csv": table_csv},
        )
