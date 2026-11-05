### Clipping and η (seed 42, each rule inside each row; columns without a prefix are rule A)

| η    | M    | min θ̂   | mean weight   | max weight   | λ̃   | step   | test DCG@5   | rare-credit   | B: test DCG@5   | B: rare-credit   |
|:-----|:-----|:--------|:--------------|:-------------|:----|:-------|:-------------|:--------------|:----------------|:-----------------|
| 0.25 | 0    | 0.190   | 1.82          | 5.3          | 3   | 2740   | 0.9307       | 0.4036        | 0.9367          | 0.4056           |
| 0.5  | 0    | 0.036   | 3.70          | 27.7         | 3   | 1010   | 0.9614       | 0.4932        | 0.9723          | 0.4382           |
| 0.5  | 0.1  | 0.036   | 3.65          | 10.0         | 3   | 980    | 0.9729       | 0.4935        | 0.9746          | 0.4673           |
| 0.5  | 0.3  | 0.036   | 2.52          | 3.3          | 3   | 2960   | 0.9222       | 0.4140        | 0.9455          | 0.3349           |
| 1    | 0    | 0.001   | 20.38         | 764.7        | 1   | 1190   | 0.8993       | 0.4785        | 0.9470          | 0.4734           |
| 1    | 0.03 | 0.001   | 13.80         | 33.3         | 1   | 540    | 0.8910       | 0.4999        | 0.9733          | 0.5160           |
| 1    | 0.1  | 0.001   | 6.66          | 10.0         | 3   | 730    | 0.9508       | 0.4759        | 0.9726          | 0.3915           |

### η variants against η = 0.5 (3 seeds, rule A)

| comparison                 | rule   | metric            | diff    | 95% CI             | relative gain   | headroom share   | seeds agree   | reading             |
|:---------------------------|:-------|:------------------|:--------|:-------------------|:----------------|:-----------------|:--------------|:--------------------|
| Rel-MF, η = 0.25 vs Rel-MF | A      | DCG@5             | -0.0388 | [-0.0504, -0.0268] | -4.0%           | -7.3%            | yes           | lower               |
| Rel-MF, η = 0.25 vs Rel-MF | A      | Rare-credit DCG@5 | -0.0801 | [-0.0966, -0.0626] | -16.6%          | -10.3%           | yes           | lower               |
| Rel-MF, η = 1 vs Rel-MF    | A      | DCG@5             | -0.0305 | [-0.0405, -0.0191] | -3.1%           | -5.8%            | yes           | lower               |
| Rel-MF, η = 1 vs Rel-MF    | A      | Rare-credit DCG@5 | -0.0016 | [-0.0146, +0.0119] | -0.3%           | -0.2%            | no            | no clear difference |

### Dimension: test DCG@5 / rare-credit DCG@5 (seed 42, rule A step)

| family   | d=30            | d=100           | d=200           |
|:---------|:----------------|:----------------|:----------------|
| Naive MF | 0.8759 / 0.2475 | 0.8944 / 0.4137 | 0.8991 / 0.4345 |
| Rel-MF   | 0.9733 / 0.4595 | 0.9729 / 0.4935 | 0.9801 / 0.4752 |

### Selected step against last step, test DCG@5 (3 seeds, rule A; BPR-MF counts epochs; R̂_n at last step is seed 42)

|                                           |   Selected step |   Last step |    Drop |   Budget (steps) |   R̂_n at last step | Selected steps   | Test peak steps   | Test peak − last   | R̂_n first < 0       |
|:------------------------------------------|----------------:|------------:|--------:|-----------------:|-------------------:|:-----------------|:------------------|:-------------------|:--------------------|
| Naive MF                                  |          0.8848 |      0.8775 |  0.0074 |             6000 |             0.0004 | 4000, 5760, 5660 | 2100, 2130, 1800  | 0.0235 to 0.0442   | never, never, never |
| BPR-MF                                    |          0.9755 |      0.9696 |  0.0060 |              100 |                    | 57, 54, 42       | 16, 9, 18         | 0.0158 to 0.0221   |                     |
| WMF                                       |          1.0030 |      1.0021 |  0.0009 |             1500 |             0.0528 | 1170, 1420, 1500 | 150, 130, 240     | 0.0058 to 0.0156   | never, never, never |
| SCAR (constant θ)                         |          1.0049 |      1.0046 |  0.0003 |             3000 |            -0.1159 | 3000, 2970, 2330 | 280, 300, 520     | 0.0034 to 0.0095   | 40, 40, 40          |
| Rel-MF                                    |          0.9732 |      0.8992 |  0.0739 |             3000 |            -0.0075 | 980, 910, 930    | 790, 790, 820     | 0.0827 to 0.0863   | 840, 840, 840       |
| Rel-MF + nnPU                             |          0.9587 |      0.9543 |  0.0044 |             3000 |             0.0007 | 2140, 2090, 2170 | 800, 790, 760     | 0.0274 to 0.0302   | 1190, 860, 870      |
| Rel-MF balanced (official-code objective) |          0.9881 |      0.9888 | -0.0007 |             3000 |            -1.2616 | 2970, 2420, 2990 | 840, 660, 1290    | 0.0037 to 0.0083   | 0, 0, 0             |

Rel-MF + nnPU minus Rel-MF at the last step, rule A winners: +0.0551 [+0.0468, +0.0630]  
rare-credit, rule A winners: -0.0103 [-0.0198, +0.0001]  
rare-credit, rule B winners: -0.0595 [-0.0717, -0.0462]  
restricted DCG@1, rule A winners: +0.0295 [+0.0155, +0.0444]  
restricted DCG@1, rule B winners: +0.0693 [+0.0516, +0.0867]

### The fits in the negative-risk figure

λ̃ = 3, M = 0, seed 42. Rel-MF: R̂_n first below zero at step 820; test peak 0.9842 at step 800, last 0.9020.
nnPU: test peak 0.9842 at step 800, last 0.9563, smallest R̂_n -0.00035.
nnPU minus Rel-MF, rule A winners, at Rel-MF's selected steps, by seed: +0.0080, -0.0004, +0.0034.
At M = 0 on both sides, λ̃ = 3, seed 42: rule A picks differ by -0.0022; against plain Rel-MF's 8 steps within 0.0005 of rule A's maximum, -0.0149 to +0.0023.

### Test peak by λ̃, Rel-MF M = 0 against nnPU (seed 42, read from the test curve)

| λ̃   | R̂_n first < 0   | Rel-MF test peak   | nnPU test peak   | nnPU − Rel-MF   |
|:----|:----------------|:-------------------|:-----------------|:----------------|
| 0.1 | 320             | 0.9627 at 440      | 0.9832 at 570    | +0.0205         |
| 0.3 | 340             | 0.9742 at 430      | 0.9842 at 550    | +0.0099         |
| 1   | 430             | 0.9795 at 500      | 0.9886 at 520    | +0.0090         |
| 3   | 820             | 0.9842 at 800      | 0.9842 at 800    | +0.0000         |
| 10  | never           | 0.8686 at 2990     | 0.8686 at 2990   | +0.0000         |

### What the raised step budgets changed (rule A, 3 seeds, same configuration, picks re-made within each budget)

| family           |   earlier budget |   current budget | steps picked, earlier   | steps picked, current   |   DCG@5, earlier |   DCG@5, current |   rare-credit, earlier |   rare-credit, current |
|:-----------------|-----------------:|-----------------:|:------------------------|:------------------------|-----------------:|-----------------:|-----------------------:|-----------------------:|
| Naive MF         |             3000 |             6000 | 2910, 3000, 3000        | 4000, 5760, 5660        |           0.9021 |           0.8848 |                 0.3751 |                 0.4385 |
| SCAR             |             1500 |             3000 | 1390, 1490, 1500        | 3000, 2970, 2330        |           1.0058 |           1.0049 |                 0.4647 |                 0.4681 |
| Rel-MF + nnPU    |             1500 |             3000 | 1430, 1500, 1360        | 2140, 2090, 2170        |           0.9613 |           0.9587 |                 0.4688 |                 0.4629 |
| Rel-MF balanced  |             1500 |             3000 | 1470, 1500, 1500        | 2970, 2420, 2990        |           0.9889 |           0.9881 |                 0.5085 |                 0.5105 |
| Rel-MF, η = 0.25 |             1500 |             3000 | 1500, 1490, 1500        | 2740, 2790, 2630        |           0.9399 |           0.9343 |                 0.3300 |                 0.4018 |

Rel-MF over naive MF, rule A, at naive MF's earlier budget (no CI): +0.0711 (+7.9%) on all items, +0.1068 on rare-credit. WMF over naive MF: +11.2%.
