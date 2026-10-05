from __future__ import annotations

import argparse
import csv
import itertools
import math
import platform
import random
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn

import build.lib.cnn as cnn
from micrograd.nn import MLP


IMPLEMENTATIONS = ("micrograd", "C", "pytorch")


@dataclass(frozen=True)
class Scenario:
    name: str
    input_size: int
    hidden_size: int
    batch_size: int
    steps: int
    learning_rate: float
    repeats: int
    implementations: tuple[str, ...]

    @property
    def layers(self) -> tuple[int, int, int]:
        return (self.input_size, self.hidden_size, 1)

    @property
    def parameter_count(self) -> int:
        return (
            self.input_size * self.hidden_size
            + self.hidden_size
            + self.hidden_size
            + 1
        )



SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        name="parity_4",
        input_size=4,
        hidden_size=2,
        batch_size=16,
        steps=10_000,
        learning_rate=0.01,
        repeats=5,
        implementations=IMPLEMENTATIONS,
    ),
    Scenario(
        name="parity_6",
        input_size=6,
        hidden_size=3,
        batch_size=32,
        steps=5_000,
        learning_rate=0.01,
        repeats=3,
        implementations=IMPLEMENTATIONS,
    ),
    Scenario(
        name="parity_8",
        input_size=8,
        hidden_size=4,
        batch_size=32,
        steps=10_000,
        learning_rate=0.01,
        repeats=3,
        implementations=("C", "pytorch"),
    ),
    Scenario(
        name="parity_10",
        input_size=10,
        hidden_size=5,
        batch_size=32,
        steps=10_000,
        learning_rate=0.01,
        repeats=2,
        implementations=("C", "pytorch"),
    ),
    Scenario(
        name="parity_12",
        input_size=12,
        hidden_size=6,
        batch_size=32,
        steps=10_000,
        learning_rate=0.01,
        repeats=2,
        implementations=("C", "pytorch"),
    ),
)

RANDOM_SEED = 20261005
WARMUP_STEPS = 100
OUTPUT_FILE = Path("parity_benchmarks.csv")



def make_parity_dataset(
    scenario: Scenario,
    seed: int,
) -> tuple[list[list[float]], list[float], int]:
    n = scenario.input_size
    all_states = list(itertools.product((-1.0, 1.0), repeat=n))

    positive = [state for state in all_states if math.prod(state) > 0]
    negative = [state for state in all_states if math.prod(state) < 0]

    if scenario.batch_size > len(all_states):
        raise ValueError(
            f"Batch size {scenario.batch_size} exceeds the {len(all_states)} "
            f"possible states for {n}-bit parity."
        )

    if scenario.batch_size % 2 != 0:
        raise ValueError("Parity benchmark batch sizes must be even.")

    rng = random.Random(seed)
    half = scenario.batch_size // 2

    selected = rng.sample(positive, half) + rng.sample(negative, half)
    rng.shuffle(selected)

    xs = [list(state) for state in selected]
    ys = [float(math.prod(state)) for state in selected]

    return xs, ys, len(all_states)



def make_torch_mlp(scenario: Scenario) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(scenario.input_size, scenario.hidden_size),
        nn.Tanh(),
        nn.Linear(scenario.hidden_size, 1),
    )


def run_micrograd(
    scenario: Scenario,
    xs: list[list[float]],
    ys: list[float],
    seed: int,
) -> float:
    random.seed(seed)
    model = MLP(scenario.input_size, [scenario.hidden_size, 1])

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


def run_c(
    scenario: Scenario,
    xs: list[list[float]],
    ys: list[float],
    seed: int,
) -> float:
    del seed
    result = cnn.run(
        len(scenario.layers),
        list(scenario.layers),
        len(xs),
        xs,
        ys,
        scenario.steps,
        scenario.learning_rate,
    )
    return float(result["loss"])


def run_pytorch(
    scenario: Scenario,
    xs: list[list[float]],
    ys: list[float],
    seed: int,
) -> float:
    torch.manual_seed(seed)
    model = make_torch_mlp(scenario)

    xs_tensor = torch.tensor(xs, dtype=torch.float32)
    ys_tensor = torch.tensor(ys, dtype=torch.float32).unsqueeze(1)

    for _ in range(scenario.steps):
        ypred = model(xs_tensor)
        loss = ((ypred - ys_tensor) ** 2).sum()

        model.zero_grad(set_to_none=True)
        loss.backward()

        with torch.no_grad():
            for parameter in model.parameters():
                parameter -= scenario.learning_rate * parameter.grad

    ypred = model(xs_tensor)
    loss = ((ypred - ys_tensor) ** 2).sum()
    return float(loss.item())


Runner = Callable[[Scenario, list[list[float]], list[float], int], float]

RUNNERS: dict[str, Runner] = {
    "micrograd": run_micrograd,
    "C": run_c,
    "pytorch": run_pytorch,
}


def configure_environment() -> None:
    """Make CPU execution explicit."""
    torch.set_num_threads(1)

    if torch.cuda.is_available():
        print("CUDA is available, but this benchmark intentionally uses CPU only.")



def warm_up(
    scenario: Scenario,
    xs: list[list[float]],
    ys: list[float],
    seed: int,
) -> None:
    """Warm up without repeating the entire expensive benchmark workload."""
    warmup_steps = min(WARMUP_STEPS, scenario.steps)
    warmup_scenario = replace(scenario, steps=warmup_steps)

    print(
        f"  Warmup ({warmup_steps} steps): "
        f"{', '.join(scenario.implementations)}"
    )

    for index, implementation in enumerate(scenario.implementations):
        RUNNERS[implementation](
            warmup_scenario,
            xs,
            ys,
            seed + index,
        )


def benchmark_scenario(
    scenario: Scenario,
    xs: list[list[float]],
    ys: list[float],
    dataset_size: int,
    rng: random.Random,
    benchmark_started_at: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    warm_up(scenario, xs, ys, seed=rng.randrange(2**32))

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
            run_seed = rng.randrange(2**32)

            start_ns = time.perf_counter_ns()
            loss = runner(scenario, xs, ys, run_seed)
            elapsed_ns = time.perf_counter_ns() - start_ns

            is_finite = math.isfinite(loss)

            rows.append(
                {
                    "benchmark_started_at": benchmark_started_at,
                    "scenario": scenario.name,
                    "implementation": implementation,
                    "replicate": replicate,
                    "order_index": order_index,
                    "execution_order": execution_order_label,
                    "input_size": scenario.input_size,
                    "hidden_size": scenario.hidden_size,
                    "output_size": 1,
                    "layers": "-".join(map(str, scenario.layers)),
                    "parameter_count": scenario.parameter_count,
                    "dataset_size": dataset_size,
                    "batch_size": scenario.batch_size,
                    "batch_coverage": scenario.batch_size / dataset_size,
                    "steps": scenario.steps,
                    "learning_rate": scenario.learning_rate,
                    "time_ns": elapsed_ns,
                    "time_s": elapsed_ns / 1_000_000_000,
                    "time_per_step_ns": elapsed_ns / scenario.steps,
                    "loss": loss,
                    "loss_is_finite": is_finite,
                    "run_seed": run_seed,
                }
            )

            print(
                f"    {implementation:10s} "
                f"{elapsed_ns / 1_000_000_000:10.4f}s "
                f"loss={loss:.6g}"
            )

    return rows



FIELDNAMES = [
    "benchmark_started_at",
    "scenario",
    "implementation",
    "replicate",
    "order_index",
    "execution_order",
    "input_size",
    "hidden_size",
    "output_size",
    "layers",
    "parameter_count",
    "dataset_size",
    "batch_size",
    "batch_coverage",
    "steps",
    "learning_rate",
    "time_ns",
    "time_s",
    "time_per_step_ns",
    "loss",
    "loss_is_finite",
    "run_seed",
]


def write_results(rows: list[dict[str, object]], output_file: Path) -> None:
    if not rows:
        raise ValueError("No benchmark results were collected.")

    with output_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark micrograd, the C library, and PyTorch on "
            "small N-bit parity networks with many iterations."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_FILE,
        help="CSV output path (default: parity_benchmarks.csv)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help=f"Dataset/order seed (default: {RANDOM_SEED})",
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
        "--steps-multiplier",
        type=float,
        default=1.0,
        help="Multiply scenario step counts by this value (default: 1.0).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Halve steps and repeats for a quick smoke test.",
    )
    parser.add_argument(
        "--micrograd-all",
        action="store_true",
        help=(
            "Also benchmark micrograd on N=8,10,12. This is off by default "
            "because scalar Python autograd can become very slow."
        ),
    )
    parser.add_argument(
        "--no-micrograd",
        action="store_true",
        help="Removes micrograd from all tests."
    )
    return parser.parse_args()


def select_scenarios(args: argparse.Namespace) -> list[Scenario]:
    if args.repeats is not None and args.repeats < 1:
        raise ValueError("--repeats must be at least 1")

    if args.steps_multiplier <= 0:
        raise ValueError("--steps-multiplier must be greater than 0")

    selected = [
        scenario
        for scenario in SCENARIOS
        if not args.scenarios or scenario.name in args.scenarios
    ]

    adjusted: list[Scenario] = []
    for scenario in selected:
        repeats = args.repeats if args.repeats is not None else scenario.repeats
        steps = max(1, round(scenario.steps * args.steps_multiplier))

        if args.quick:
            repeats = max(1, repeats // 2)
            steps = max(1, steps // 2)

        implementations = scenario.implementations
        if args.micrograd_all:
            implementations = IMPLEMENTATIONS

        if args.no_micrograd:
            implementations = ("C", "pytorch")

        adjusted.append(
            replace(
                scenario,
                steps=steps,
                repeats=repeats,
                implementations=implementations,
            )
        )

    return adjusted


def main() -> None:
    args = parse_args()
    scenarios = select_scenarios(args)
    configure_environment()

    benchmark_started_at = datetime.now(timezone.utc).isoformat()
    rng = random.Random(args.seed)

    print("Starting parity benchmarks...")
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Platform: {platform.platform()}")
    print(f"PyTorch CPU threads: {torch.get_num_threads()}")
    print(f"Seed: {args.seed}")
    print()

    all_rows: list[dict[str, object]] = []

    for scenario in scenarios:
        xs, ys, dataset_size = make_parity_dataset(
            scenario,
            seed=args.seed + scenario.input_size,
        )

        print(
            f"{scenario.name}: "
            f"layers={scenario.layers}, "
            f"params={scenario.parameter_count}, "
            f"batch={scenario.batch_size}/{dataset_size}, "
            f"steps={scenario.steps}, "
            f"repeats={scenario.repeats}"
        )

        scenario_rows = benchmark_scenario(
            scenario,
            xs,
            ys,
            dataset_size,
            rng,
            benchmark_started_at,
        )
        all_rows.extend(scenario_rows)

        write_results(all_rows, args.output)
        print(f"  Saved {len(all_rows)} measurements to {args.output}\n")

    print("Parity benchmark complete.")
    print(f"Measurements: {len(all_rows)}")
    print(f"Output written to: {args.output.resolve()}")


if __name__ == "__main__":
    main()