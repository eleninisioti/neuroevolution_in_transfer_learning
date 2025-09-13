# Cumulative Fitness Results (All Steps) with Statistical Analysis

Cumulative fitness by summing all fitness values across all training steps.

## Acrobot

### Cumulative Fitness Results

| Method | Cumulative Fitness (All Steps) | Std Dev |
|--------|-----------------------------------|----------|
| PPO | -346124.32 | 24186.78 |
| GA | -128614.20 | 3312.54 |
| OpenES | -198205.70 | 28079.14 |
| CMA-ES | -134384.00 | 8743.38 |

### Group-Level Statistical Tests

| Test | Statistic | p-value | Significant (α=0.05) |
|------|-----------|---------|---------------------|
| ANOVA | F=252.538 | 0.000 | Yes |
| Kruskal-Wallis | H=32.775 | 0.000 | Yes |

### Pairwise Comparisons

| Method 1 | Method 2 | Test | Statistic | p-value | p-value (Bonferroni) | Significant |
|----------|----------|------|-----------|---------|---------------------|-------------|
| PPO | GA | Mann-Whitney U | 0.000 | 0.000 | 0.001 | Yes |
| PPO | OpenES | t-test | -11.974 | 0.000 | 0.000 | Yes |
| PPO | CMA-ES | Mann-Whitney U | 0.000 | 0.000 | 0.001 | Yes |
| GA | OpenES | Mann-Whitney U | 99.000 | 0.000 | 0.001 | Yes |
| GA | CMA-ES | Mann-Whitney U | 71.000 | 0.121 | 0.727 | No |
| OpenES | CMA-ES | Mann-Whitney U | 3.000 | 0.000 | 0.003 | Yes |

## Cartpole

### Cumulative Fitness Results

| Method | Cumulative Fitness (All Steps) | Std Dev |
|--------|-----------------------------------|----------|
| PPO | 79326.97 | 30141.14 |
| GA | 285272.30 | 45453.70 |
| OpenES | 225220.80 | 32580.03 |
| CMA-ES | 341348.50 | 37262.92 |

### Group-Level Statistical Tests

| Test | Statistic | p-value | Significant (α=0.05) |
|------|-----------|---------|---------------------|
| ANOVA | F=84.389 | 0.000 | Yes |
| Kruskal-Wallis | H=32.087 | 0.000 | Yes |

### Pairwise Comparisons

| Method 1 | Method 2 | Test | Statistic | p-value | p-value (Bonferroni) | Significant |
|----------|----------|------|-----------|---------|---------------------|-------------|
| PPO | GA | Mann-Whitney U | 0.000 | 0.000 | 0.001 | Yes |
| PPO | OpenES | Mann-Whitney U | 0.000 | 0.000 | 0.001 | Yes |
| PPO | CMA-ES | Mann-Whitney U | 0.000 | 0.000 | 0.001 | Yes |
| GA | OpenES | t-test | 3.221 | 0.005 | 0.028 | Yes |
| GA | CMA-ES | t-test | -2.862 | 0.010 | 0.062 | No |
| OpenES | CMA-ES | t-test | -7.038 | 0.000 | 0.000 | Yes |

## Mountaincar

### Cumulative Fitness Results

| Method | Cumulative Fitness (All Steps) | Std Dev |
|--------|-----------------------------------|----------|
| PPO | -388544.77 | 1409.84 |
| GA | -369260.10 | 9580.04 |
| OpenES | -358137.60 | 18168.02 |
| CMA-ES | -386806.40 | 11648.98 |

### Group-Level Statistical Tests

| Test | Statistic | p-value | Significant (α=0.05) |
|------|-----------|---------|---------------------|
| ANOVA | F=13.688 | 0.000 | Yes |
| Kruskal-Wallis | H=25.001 | 0.000 | Yes |

### Pairwise Comparisons

| Method 1 | Method 2 | Test | Statistic | p-value | p-value (Bonferroni) | Significant |
|----------|----------|------|-----------|---------|---------------------|-------------|
| PPO | GA | Mann-Whitney U | 0.000 | 0.000 | 0.001 | Yes |
| PPO | OpenES | Mann-Whitney U | 0.000 | 0.000 | 0.001 | Yes |
| PPO | CMA-ES | Mann-Whitney U | 40.000 | 0.473 | 1.000 | No |
| GA | OpenES | t-test | -1.625 | 0.122 | 0.730 | No |
| GA | CMA-ES | t-test | 3.490 | 0.003 | 0.016 | Yes |
| OpenES | CMA-ES | t-test | 3.985 | 0.001 | 0.005 | Yes |

