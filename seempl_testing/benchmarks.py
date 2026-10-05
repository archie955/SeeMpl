from __future__ import annotations

import argparse
import csv
import platform
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd
import torch
import torch.nn as nn

import build.lib.cnn as cnn
from micrograd.nn import MLP


TINY_IMPLEMENTATIONS = ("micrograd", "C", "pytorch")
IMPLEMENTATIONS = ("C", "pytorch")


@dataclass(frozen=True)
class Scenario:
    name: str
    layers: tuple[int, ...]
    batch_size: int
    steps: int
    learning_rate: float
    repeats: int
    implementations: tuple[str, ...]


SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        name="tiny",
        layers=(3, 4, 4, 1),
        batch_size=4,
        steps=5_000,
        learning_rate=0.01,
        repeats=30,
        implementations=TINY_IMPLEMENTATIONS
    ),
    Scenario(
        name="small",
        layers=(8, 16, 16, 1),
        batch_size=32,
        steps=2_000,
        learning_rate=0.01,
        repeats=20,
        implementations=IMPLEMENTATIONS
    ),
    Scenario(
        name="medium",
        layers=(32, 64, 64, 1),
        batch_size=128,
        steps=1_000,
        learning_rate=0.01,
        repeats=10,
        implementations=IMPLEMENTATIONS
    ),
    Scenario(
        name="large",
        layers=(64, 128, 128, 1),
        batch_size=512,
        steps=250,
        learning_rate=0.01,
        repeats=5,
        implementations=IMPLEMENTATIONS
    ),
    Scenario(
        name="deep",
        layers=(32, 64, 64, 64, 64, 32, 1),
        batch_size=128,
        steps=1_000,
        learning_rate=0.01,
        repeats=10,
        implementations=IMPLEMENTATIONS
    ),
)

RANDOM_SEED = 20261005
WARMUP_RUNS = 1
OUTPUT_FILE = Path("benchmarks.csv")



def make_dataset(scenario: Scenario, seed: int) -> tuple[list[list[float]], list[float]]:
    rng = random.Random(seed)
    input_size = scenario.layers[0]

    xs = [
        [rng.uniform(-1.0, 1.0) for _ in range(input_size)]
        for _ in range(scenario.batch_size)
    ]

    ys = []
    for x in xs:
        weighted_sum = sum((i + 1) * value for i, value in enumerate(x))
        target = weighted_sum / input_size
        ys.append(float(target))

    return xs, ys


def make_torch_mlp(layers: tuple[int, ...]) -> nn.Sequential:
    modules: list[nn.Module] = []

    for input_size, output_size in zip(layers, layers[1:]):
        modules.append(nn.Linear(input_size, output_size))
        if output_size != layers[-1]:
            modules.append(nn.Tanh())

    return nn.Sequential(*modules)


def run_micrograd(scenario: Scenario, xs: list[list[float]], ys: list[float]) -> float:
    model = MLP(scenario.layers[0], list(scenario.layers[1:]))

    for _ in range(scenario.steps):
        ypred = [model(x) for x in xs]
        loss = sum(
            (yout - ygt) ** 2
            for ygt, yout in zip(ys, ypred)
        )

        model.zero_grad()
        loss.backward()

        for parameter in model.parameters():
            parameter.data -= scenario.learning_rate * parameter.grad

    ypred = [model(x) for x in xs]
    loss = sum(
        (yout - ygt) ** 2
        for ygt, yout in zip(ys, ypred)
    )
    return float(loss.data)


def run_c(scenario: Scenario, xs: list[list[float]], ys: list[float]) -> float:
    result = cnn.run(
        len(scenario.layers),
        list(scenario.layers),
        scenario.batch_size,
        xs,
        ys,
        scenario.steps,
        scenario.learning_rate,
    )
    return float(result["loss"])


def run_pytorch(scenario: Scenario, xs: list[list[float]], ys: list[float]) -> float:
    model = make_torch_mlp(scenario.layers)

    xs_tensor = torch.tensor(xs, dtype=torch.float32)
    ys_tensor = torch.tensor(ys, dtype=torch.float32).unsqueeze(1)

    for _ in range(scenario.steps):
        ypred = model(xs_tensor)
        loss = ((ypred - ys_tensor) ** 2).sum()

        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            for parameter in model.parameters():
                parameter -= scenario.learning_rate * parameter.grad

    ypred = model(xs_tensor)
    loss = ((ypred - ys_tensor) ** 2).sum()
    return float(loss.item())


RUNNERS: dict[str, Callable[[Scenario, list[list[float]], list[float]], float]] = {
    "micrograd": run_micrograd,
    "C": run_c,
    "pytorch": run_pytorch,
}


def configure_environment() -> None:
    torch.set_num_threads(1)

    if torch.cuda.is_available():
        print("CUDA is available, but this benchmark intentionally uses CPU only.")


def warm_up(
    scenario: Scenario,
    datasets: tuple[list[list[float]], list[float]],
) -> None:
    xs, ys = datasets

    print(f"Warming up: {scenario.name}")
    for implementation in scenario.implementations:
        for _ in range(WARMUP_RUNS):
            RUNNERS[implementation](scenario, xs, ys)


def benchmark_scenario(
    scenario: Scenario,
    datasets: tuple[list[list[float]], list[float]],
    rng: random.Random,
    benchmark_started_at: str,
) -> list[dict[str, object]]:
    xs, ys = datasets
    rows: list[dict[str, object]] = []

    warm_up(scenario, datasets)

    for replicate in range(1, scenario.repeats + 1):
        execution_order = list(scenario.implementations)
        rng.shuffle(execution_order)
        execution_order_label = ">".join(execution_order)

        print(
            f"{scenario.name}: replicate {replicate}/{scenario.repeats} "
            f"[{execution_order_label}]"
        )

        for order_index, implementation in enumerate(execution_order, start=1):
            runner = RUNNERS[implementation]

            start_ns = time.perf_counter_ns()
            loss = runner(scenario, xs, ys)
            elapsed_ns = time.perf_counter_ns() - start_ns

            rows.append(
                {
                    "benchmark_started_at": benchmark_started_at,
                    "scenario": scenario.name,
                    "implementation": implementation,
                    "replicate": replicate,
                    "order_index": order_index,
                    "execution_order": execution_order_label,
                    "layers": "-".join(map(str, scenario.layers)),
                    "input_size": scenario.layers[0],
                    "output_size": scenario.layers[-1],
                    "batch_size": scenario.batch_size,
                    "steps": scenario.steps,
                    "learning_rate": scenario.learning_rate,
                    "time_ns": elapsed_ns,
                    "time_s": elapsed_ns / 1_000_000_000,
                    "loss": loss,
                }
            )

    return rows


def write_results(rows: list[dict[str, object]], output_file: Path) -> None:
    if not rows:
        raise ValueError("No benchmark results were collected.")

    fieldnames = list(rows[0].keys())

    with output_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_run_summary(rows: list[dict[str, object]], output_file: Path) -> None:
    df = pd.DataFrame(rows)

    print("\nBenchmark complete.")
    print(f"Measurements: {len(df)}")
    print(f"CSV columns: {', '.join(df.columns)}")
    print(f"Output written to: {output_file.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark micrograd, C, and PyTorch.")
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_FILE,
        help="CSV output path (default: benchmarks.csv)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help=f"Seed used for data generation and execution-order shuffling (default: {RANDOM_SEED})",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        dest="scenarios",
        choices=[scenario.name for scenario in SCENARIOS],
        help="Run only the named scenario; may be supplied multiple times.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=None,
        help="Override repeats for every selected scenario.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Halve steps and repeats for a quick smoke test.",
    )
    return parser.parse_args()


def select_scenarios(args: argparse.Namespace) -> list[Scenario]:
    selected = [
        scenario
        for scenario in SCENARIOS
        if not args.scenarios or scenario.name in args.scenarios
    ]

    if args.repeats is not None and args.repeats < 1:
        raise ValueError("--repeats must be at least 1")

    adjusted: list[Scenario] = []
    for scenario in selected:
        repeats = args.repeats if args.repeats is not None else scenario.repeats
        steps = scenario.steps

        if args.quick:
            repeats = max(1, repeats // 2)
            steps = max(1, steps // 2)

        adjusted.append(
            Scenario(
                name=scenario.name,
                layers=scenario.layers,
                batch_size=scenario.batch_size,
                steps=steps,
                learning_rate=scenario.learning_rate,
                repeats=repeats,
                implementations=scenario.implementations
            )
        )

    return adjusted


def main() -> None:
    args = parse_args()
    scenarios = select_scenarios(args)

    configure_environment()

    benchmark_started_at = datetime.now(timezone.utc).isoformat()
    rng = random.Random(args.seed)

    print("Starting benchmarks...")
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Platform: {platform.platform()}")
    print(f"PyTorch CPU threads: {torch.get_num_threads()}")
    print(f"Seed: {args.seed}")
    print()

    all_rows: list[dict[str, object]] = []

    for scenario in scenarios:
        datasets = make_dataset(scenario, seed=args.seed)
        scenario_rows = benchmark_scenario(
            scenario,
            datasets,
            rng,
            benchmark_started_at,
        )
        all_rows.extend(scenario_rows)

    write_results(all_rows, args.output)

    print_run_summary(all_rows, args.output)


if __name__ == "__main__":
    main()