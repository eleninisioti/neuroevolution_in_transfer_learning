# Cumulative Fitness Results with Statistical Analysis

Cumulative fitness by summing fitness values at steps 199, 399, 599, 799, 999, 1199, 1399, 1599, 1799, 1999.

## Acrobot

### Cumulative Fitness Results

| Method | Cumulative Fitness (Specific Steps) | Std Dev |
|--------|-----------------------------------|----------|
| PPO | -1661.60 | 202.92 |
| GA | -638.20 | 15.56 |
| OpenES | -932.20 | 126.55 |
| CMA-ES | -1437.00 | 211.36 |

### Group-Level Statistical Tests

| Test | Statistic | p-value | Significant (α=0.05) |
|------|-----------|---------|---------------------|
| ANOVA | F=76.662 | 0.000 | Yes |
| Kruskal-Wallis | H=33.852 | 0.000 | Yes |

### Pairwise Comparisons

| Method 1 | Method 2 | Test | Statistic | p-value | p-value (Bonferroni) | Significant |
|----------|----------|------|-----------|---------|---------------------|-------------|
| PPO | GA | t-test | -15.086 | 0.000 | 0.000 | Yes |
| PPO | OpenES | t-test | -9.150 | 0.000 | 0.000 | Yes |
| PPO | CMA-ES | t-test | -2.300 | 0.034 | 0.202 | No |
| GA | OpenES | t-test | 6.918 | 0.000 | 0.000 | Yes |
| GA | CMA-ES | t-test | 11.307 | 0.000 | 0.000 | Yes |
| OpenES | CMA-ES | t-test | 6.147 | 0.000 | 0.000 | Yes |

## Cartpole

### Cumulative Fitness Results

| Method | Cumulative Fitness (Specific Steps) | Std Dev |
|--------|-----------------------------------|----------|
| PPO | 464.19 | 92.73 |
| GA | 1481.30 | 210.50 |
| OpenES | 1188.10 | 161.58 |
| CMA-ES | 670.40 | 228.30 |

### Group-Level Statistical Tests

| Test | Statistic | p-value | Significant (α=0.05) |
|------|-----------|---------|---------------------|
| ANOVA | F=59.767 | 0.000 | Yes |
| Kruskal-Wallis | H=32.312 | 0.000 | Yes |

### Pairwise Comparisons

| Method 1 | Method 2 | Test | Statistic | p-value | p-value (Bonferroni) | Significant |
|----------|----------|------|-----------|---------|---------------------|-------------|
| PPO | GA | t-test | -13.265 | 0.000 | 0.000 | Yes |
| PPO | OpenES | t-test | -11.657 | 0.000 | 0.000 | Yes |
| PPO | CMA-ES | t-test | -2.511 | 0.022 | 0.131 | No |
| GA | OpenES | t-test | 3.315 | 0.004 | 0.023 | Yes |
| GA | CMA-ES | t-test | 7.834 | 0.000 | 0.000 | Yes |
| OpenES | CMA-ES | t-test | 5.553 | 0.000 | 0.000 | Yes |

## Mountaincar

### Cumulative Fitness Results

| Method | Cumulative Fitness (Specific Steps) | Std Dev |
|--------|-----------------------------------|----------|
| PPO | -1916.63 | 10.07 |
| GA | -1769.80 | 63.34 |
| OpenES | -1709.90 | 95.11 |
| CMA-ES | -2000.00 | 0.00 |

### Group-Level Statistical Tests

| Test | Statistic | p-value | Significant (α=0.05) |
|------|-----------|---------|---------------------|
| ANOVA | F=48.330 | 0.000 | Yes |
| Kruskal-Wallis | H=34.039 | 0.000 | Yes |

### Pairwise Comparisons

| Method 1 | Method 2 | Test | Statistic | p-value | p-value (Bonferroni) | Significant |
|----------|----------|------|-----------|---------|---------------------|-------------|
| PPO | GA | Mann-Whitney U | 0.000 | 0.000 | 0.001 | Yes |
| PPO | OpenES | Mann-Whitney U | 0.000 | 0.000 | 0.001 | Yes |
| PPO | CMA-ES | Mann-Whitney U | 100.000 | 0.000 | 0.000 | Yes |
| GA | OpenES | t-test | -1.573 | 0.133 | 0.799 | No |
| GA | CMA-ES | t-test | 10.904 | 0.000 | 0.000 | Yes |
| OpenES | CMA-ES | t-test | 9.151 | 0.000 | 0.000 | Yes |

