### Coat, directional only (rule A unless prefixed with B; 3 seeds)

|                                           |   DCG@5 |   Recall@5 |   AP@5 |   Rare@5 |   B: DCG@5 |   B: Rare@5 |
|:------------------------------------------|--------:|-----------:|-------:|---------:|-----------:|------------:|
| Random (analytic)                         |  0.6982 |     0.3398 | 0.2216 |   0.4056 |            |             |
| Popularity                                |  0.9627 |     0.4828 | 0.3514 |   0.0184 |     0.9627 |      0.0184 |
| Naive MF                                  |  0.8504 |     0.4208 | 0.2970 |   0.3055 |     0.9359 |      0.1366 |
| BPR-MF                                    |  0.9949 |     0.5101 | 0.3888 |   0.3046 |     0.9793 |      0.3104 |
| EASE                                      |  0.9655 |     0.4871 | 0.3619 |   0.2058 |     0.9494 |      0.2081 |
| WMF                                       |  1.0351 |     0.5394 | 0.4007 |   0.1901 |     1.0267 |      0.2040 |
| SCAR (constant θ)                         |  1.0050 |     0.5060 | 0.3895 |   0.3259 |     1.0426 |      0.1882 |
| Rel-MF                                    |  0.9570 |     0.4964 | 0.3605 |   0.2691 |     0.9990 |      0.2695 |
| Rel-MF + nnPU                             |  0.9615 |     0.4897 | 0.3583 |   0.2872 |     0.9990 |      0.2695 |
| Rel-MF balanced (official-code objective) |  0.9168 |     0.4747 | 0.3354 |   0.3080 |     0.9592 |      0.2975 |
| Ideal (analytic)                          |  1.9784 |     0.9331 | 1.0000 |   1.5374 |            |             |

### Paired differences

| comparison                                          | rule   | metric            | diff    | 95% CI             | relative gain   | headroom share   | seeds agree   | reading             |
|:----------------------------------------------------|:-------|:------------------|:--------|:-------------------|:----------------|:-----------------|:--------------|:--------------------|
| Rel-MF vs Naive MF                                  | A      | DCG@5             | +0.1066 | [+0.0558, +0.1625] | +12.5%          | +9.4%            | yes           | higher              |
| Rel-MF vs Naive MF                                  | A      | Rare-credit DCG@5 | -0.0364 | [-0.0918, +0.0207] | -11.9%          | -3.0%            | yes           | no clear difference |
| Rel-MF vs Naive MF                                  | B      | DCG@5             | +0.0631 | [+0.0035, +0.1162] | +6.7%           | +6.1%            | yes           | higher              |
| Rel-MF vs Naive MF                                  | B      | Rare-credit DCG@5 | +0.1329 | [+0.0873, +0.1853] | +97.4%          | +9.5%            | yes           | higher              |
| Rel-MF + nnPU vs Rel-MF                             | A      | DCG@5             | +0.0045 | [-0.0226, +0.0330] | +0.5%           | +0.4%            | no            | no clear difference |
| Rel-MF + nnPU vs Rel-MF                             | A      | Rare-credit DCG@5 | +0.0181 | [-0.0145, +0.0530] | +6.7%           | +1.4%            | yes           | no clear difference |
| Rel-MF + nnPU vs Rel-MF                             | B      | DCG@5             | +0.0000 | [+0.0000, +0.0000] | +0.0%           | +0.0%            | yes           | no clear difference |
| Rel-MF + nnPU vs Rel-MF                             | B      | Rare-credit DCG@5 | +0.0000 | [+0.0000, +0.0000] | +0.0%           | +0.0%            | yes           | no clear difference |
| Rel-MF balanced (official-code objective) vs Rel-MF | A      | DCG@5             | -0.0401 | [-0.0866, +0.0068] | -4.2%           | -3.9%            | yes           | no clear difference |
| Rel-MF balanced (official-code objective) vs Rel-MF | A      | Rare-credit DCG@5 | +0.0389 | [-0.0049, +0.0829] | +14.5%          | +3.1%            | yes           | no clear difference |
| Rel-MF balanced (official-code objective) vs Rel-MF | B      | DCG@5             | -0.0398 | [-0.0929, +0.0221] | -4.0%           | -4.1%            | no            | no clear difference |
| Rel-MF balanced (official-code objective) vs Rel-MF | B      | Rare-credit DCG@5 | +0.0280 | [-0.0161, +0.0741] | +10.4%          | +2.2%            | yes           | no clear difference |
| Popularity vs Rel-MF                                | A      | DCG@5             | +0.0058 | [-0.0732, +0.0798] | +0.6%           | +0.6%            | no            | no clear difference |
| Popularity vs Rel-MF                                | A      | Rare-credit DCG@5 | -0.2507 | [-0.3075, -0.1987] | -93.2%          | -19.8%           | yes           | lower               |
| Popularity vs Rel-MF                                | B      | DCG@5             | -0.0363 | [-0.1129, +0.0371] | -3.6%           | -3.7%            | no            | no clear difference |
| Popularity vs Rel-MF                                | B      | Rare-credit DCG@5 | -0.2511 | [-0.3186, -0.1940] | -93.2%          | -19.8%           | yes           | lower               |
