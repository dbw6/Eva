import argparse

from simulator.runner import build_runner_config, run


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="EVA unified simulator runner")
    parser.add_argument("--study", type=str, default="fig9_fc", help="Named study preset to execute.")
    parser.add_argument("--output-dir", type=str, default="simulator/output", help="Base output directory.")
    parser.add_argument("--models", type=str, default=None, help="Comma-separated model overrides.")
    parser.add_argument("--methods", type=str, default=None, help="Comma-separated method overrides.")
    parser.add_argument("--scenarios", type=str, default=None, help="Comma-separated scenario overrides for end-to-end studies.")
    parser.add_argument("--sequence-lengths", type=str, default=None, help="Comma-separated sequence lengths.")
    parser.add_argument("--batch-sizes", type=str, default=None, help="Comma-separated batch sizes.")
    parser.add_argument("--phase", type=str, default=None, help="Optional phase override.")
    parser.add_argument("--ops-mode", type=str, default=None, help="Optional ops mode override.")
    parser.add_argument("--execution-mode", type=str, default=None, help="Optional execution mode override, such as sample or full.")
    parser.add_argument("--mem-width", type=int, default=1024, help="Memory width in bits for bandwidth modeling.")
    parser.add_argument("--vq-array-height", type=int, default=32, help="EVA decode array height.")
    parser.add_argument("--vq-array-width", type=int, default=8, help="EVA decode array width.")
    parser.add_argument("--vq-adder-tree-size", type=int, default=None, help="Optional EVA adder-tree size override.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    config = build_runner_config(
        study=args.study,
        output_dir=args.output_dir,
        models=args.models,
        methods=args.methods,
        scenario_names=args.scenarios,
        sequence_lengths=args.sequence_lengths,
        batch_sizes=args.batch_sizes,
        phase=args.phase,
        ops_mode=args.ops_mode,
        execution_mode=args.execution_mode,
        mem_width=args.mem_width,
        vq_array_height=args.vq_array_height,
        vq_array_width=args.vq_array_width,
        vq_adder_tree_size=args.vq_adder_tree_size,
    )
    artifacts = run(config)
    print(f"Results written to {artifacts.output_dir}")
    print(f"Cycles CSV: {artifacts.cycles_csv}")
    print(f"Energy CSV: {artifacts.energy_csv}")
    print(f"Power CSV: {artifacts.power_csv}")
    if artifacts.aggregated_csv is not None:
        print(f"Aggregated CSV: {artifacts.aggregated_csv}")
    if artifacts.verification_json is not None:
        print(f"Verification JSON: {artifacts.verification_json}")
    for label, path in sorted(artifacts.reports.items()):
        if path == artifacts.verification_json:
            continue
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
