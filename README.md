# evo-tom

Simulation and analysis code for the paper "Environmental Requirements for the Evolution of Theory of Mind".

## Prerequisites

- Install [ghc](https://www.haskell.org/ghc/) 8.10.4, [cabal-install](https://www.haskell.org/cabal/) 3.4.\*, and [protoc](https://github.com/protocolbuffers/protobuf) 3.17.\*.
- Clone this repo.

## Tournament

The goal of running the tournaments is to get the pairwise mean payoff matrices. Run

``` shell
cabal run tournament -- [environment]
```

where `[environment]` can be `os-pf`, `os-ps`, `os-pv`, `ov-pf`, `ov-ps`, or `ov-pv`. For example, if you run

``` shell
cabal run tournament -- os-pf
```

the simulation will be run for the OS-PF environment and the data will be written to `/output/os-pf.bin`. This file is in the [Protocol Buffers](https://developers.google.com/protocol-buffers) binary format, and its specification is in `/results.proto`. The pairwise mean payoff matrix, the standard error of each of the entries (the SEM matrix), and the computation time will be printed in the console. The agents corresponding to the rows and columns in the matrices are, in the following order,

1. random agent
2. tit-for-tat agent
3. naïve utility maximizer (λ = -1)
4. naïve utility maximizer (λ = 0)
5. naïve utility maximizer (λ = 1)
6. fixed belief maximizer (λ = -1)
7. fixed belief maximizer (λ = 0)
8. fixed belief maximizer (λ = 1)
9. reinforcement-learning agent (λ = -1)
10. reinforcement-learning agent (λ = 0)
11. reinforcement-learning agent (λ = 1)
12. theory-of-mind agent (λ = -1)
13. theory-of-mind agent (λ = 0)
14. theory-of-mind agent (λ = 1)

The following table shows the number of repetitions of each supergame (see definition in SI of the paper) and the total number of rounds in the simulation of each environment:

| Environment | # of repetitions | Total # of rounds |
|-------------|------------------|-------------------|
| OS-PF       | 100              | 100×100×105       |
| OS-PS       | 10,000           | 100×10,000×105    |
| OS-PV       | 100              | 100×100×105       |
| OV-PF       | 500              | 100×24×500        |
| OV-PS       | 50,000           | 100×24×50,000     |
| OV-PV       | 50,000           | 100×24×50,000     |

The "105" in the OS environments stems from the fact that there are 105 unique pairs of agents. The "24" in the OV environments stems from the fact that there are 24 agents in either of the two groups (see SI of the paper).

After you have run the simulation for all the 6 environments (i.e., the 6 binary files have to be in place), you can write the pairwise mean payoff matrices and the SEM matrices to files by running

``` shell
cabal run analysis -- [command]
```

where `[command]` can be either `writeMeans` or `writeSems`.

## Evolution

To run the evolutionary simulation based on the pairwise mean payoff matrices, execute

``` shell
cabal run evo -- [environment]
```

where `[environment]` is one of the 6 environments as above. The data and plots will be written to `/output/evo-[environment].txt` and `/output/evo-[environment].pdf`. You can also adjust the logging interval (default is 2) by changing the `interval` variable in `evo.hs`.

## What's included in this repo

The protobuf binary files generated from the tournaments are **not** included because they are quite large (≈200MB total). All other output files are included in `/output/`.

## If something goes wrong...

Please [submit an issue](https://github.com/jameswhqi/evo-tom/issues) or [contact me](mailto:wqi@ucsd.edu) if you encounter any problem when using the code.
