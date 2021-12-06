# evo-tom

Code and data for the paper "The Evolution of Theory of Mind on Welfare Tradeoff Ratios".

## Simulation

Everything in this section happens under the `simulation/` directory.

### Prerequisites

Install [stack](https://docs.haskellstack.org/en/stable/README/) and [protoc](https://github.com/protocolbuffers/protobuf).

### Tournament

The goal of running the tournaments is to get the pairwise mean payoff matrices. Run

``` shell
stack run tournament -- [environment]
```

where `[environment]` can be `os-pf`, `os-ps`, `os-pv`, `ov-pf`, `ov-ps`, or `ov-pv`. For example, if you run

``` shell
stack run tournament -- os-pf
```

the simulation will be run for the OS-PF environment and the data will be written to `output/os-pf.bin`. This file is in the [Protocol Buffers](https://developers.google.com/protocol-buffers) binary format, and its specification is in `results.proto`. The pairwise mean payoff matrix, the standard error of each of the entries (the SEM matrix), and the computation time will be printed in the console. The agents corresponding to the rows and columns in the matrices are, in the following order,

1. random agent
2. tit-for-tat agent
3. reinforcement-learning agent (λ = -1)
4. reinforcement-learning agent (λ = 0)
5. reinforcement-learning agent (λ = 1)
6. naïve utility maximizer (λ = -1)
7. naïve utility maximizer (λ = 0)
8. naïve utility maximizer (λ = 1)
9. fixed belief maximizer (λ = -1)
10. fixed belief maximizer (λ = 0)
11. fixed belief maximizer (λ = 1)
12. theory-of-mind agent (λ = -1)
13. theory-of-mind agent (λ = 0)
14. theory-of-mind agent (λ = 1)

The following table shows the number of repetitions of supergames (see definition in the paper) and the total number of rounds in the simulation of each environment:

| Environment | # of repetitions | Total # of rounds |
|-------------|------------------|-------------------|
| OS-PF       | 100              | 100×100×105       |
| OS-PS       | 10,000           | 100×10,000×105    |
| OS-PV       | 100              | 100×100×105       |
| OV-PF       | 500              | 100×24×500        |
| OV-PS       | 50,000           | 100×24×50,000     |
| OV-PV       | 50,000           | 100×24×50,000     |

The "105" in the OS environments stems from the fact that there are 105 unique pairs of agents. The "24" in the OV environments stems from the fact that there are 24 agents in either of the two groups (see paper).

After you have run the simulation for all the 6 environments (i.e., the 6 binary files have to be in place), you can write the pairwise mean payoff matrices and the SEM matrices to files by running

``` shell
stack run analysis -- [command]
```

where `[command]` can be either `writeMeans` or `writeSems`.

### Evolution

To run the evolutionary simulation starting from equal proportions of the agents based on the pairwise mean payoff matrices, execute

``` shell
stack run evo -- process
```

The data and plots will be written to `output/evo-[environment].txt` and `output/evo-[environment].pdf`.

### What's included in this repo

The protobuf binary files generated from the tournaments are **not** included because they are quite large (≈200MB total). All other output files are included in `output/`.

## Experiment 1

Everything in this section happens under the `experiment1/` directory.

### Data

Raw data can be found under `data/`.

### Analysis

To run the analysis, install Python 3, [NumPy](https://numpy.org/), and [Funcy](https://github.com/Suor/funcy) (the newest versions should work). Then run

```shell
python3 analysis.py
```

## Experiment 2

Everything in this section happens under the `experiment2/` directory.

### Data

Raw data can be found under `data/`.

### Analysis

To run the analysis, install Python 3, NumPy, Funcy, [CmdStanPy](https://github.com/stan-dev/cmdstanpy) (including [CmdStan](https://mc-stan.org/users/interfaces/cmdstan)), and [statsmodels](https://www.statsmodels.org/) (the newest versions should work). Then run

```shell
python3 analysis.py
```

## If something goes wrong...

Please [submit an issue](https://github.com/jameswhqi/evo-tom/issues) or [contact me](mailto:wqi@ucsd.edu) if you encounter any problem when using the code.
