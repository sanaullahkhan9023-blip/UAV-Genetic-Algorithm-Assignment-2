# Assignment 2 - UAV Genetic Algorithm Strategy Engine

Bahria University, Lahore Campus - Artificial Intelligence, BSCS 5.

This is a self-contained educational simulation, not flight-ready or certified drone software.

## Setup

Python 3.10+ recommended.

```bash
pip install -r requirements.txt
python strategy_engine.py --output outputs --seed 12 --population 18 --generations 20 --mutation 0.14
```

The entrypoint writes `fitness_history.csv` and `fitness_history.png`, three generations of route + sensor maps, a random-search baseline map, `summary.txt`, and `sensitivity.csv`.

## How it works

- 30 x 30 cells; 3 drones; 4 scan stops per drone; all missions begin and end at base (1,1).
- Encoded genes: continuous 2D waypoint positions, speed 0.8-2.0 simulation units, and integer sensor levels 0/1/2.
- High-priority cells and hazardous regions have synthetic intensity maps. Black boundaries show hard no-fly rectangles.
- Repairs snap forbidden waypoint locations to free cells and route via selected corridors when direct segments are blocked. Unresolved intersections and exceeding route/time constraints incur a large penalty.
- Scores include priority-weighted unique footprint coverage and priority-weighted probabilistic detection, offset by normalized mission energy, time, and hazard exposure.
- 3-way tournament selection, drone-segment crossover plus geometric blend, mixed Gaussian/relocation/speed/sensor mutation, and elitism.
- Stops when it reaches the user-specified generation limit.
- Random-search comparison gets `population * (generations + 1)` evaluations, matching the GA total.
- The sensitivity experiments vary population, mutation rate, and tournament size one at a time; their budgets are not equal because population changes, so interpret them as settings comparisons, not evaluation-budget-adjusted causal proof.

## Repository layout

```
strategy_engine.py
requirements.txt
README.md
outputs/
technical_report.pdf
```

## Reproducibility

The default random seed is 12; results and images are deterministic with a fixed NumPy/Matplotlib environment. Update `--seed` to test other stochastic runs. Numeric time, hazard, energy, and detection are simplified mathematical proxies, not measured flight performance.

## Method caveats

- Corridor repairs are heuristic and are not guaranteed to solve every obstacle configuration. Residual intersections are always penalized.
- Flight duration is a fleet aggregate proxy, and sensor scan timing is assumed independent of coverage evolution.
- For production research, evaluate many seeds, track non-dominated Pareto fronts, validate a physics-based energy model, and incorporate changing weather/time dynamics.
