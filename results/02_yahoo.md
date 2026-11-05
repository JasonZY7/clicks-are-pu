### Main table @5, Yahoo! R3 (rule A unless prefixed with B; 3 seeds)

|                                           |   DCG@5 |   Recall@5 |   AP@5 |   Rare@5 |   B: DCG@5 |   B: Rare@5 |
|:------------------------------------------|--------:|-----------:|-------:|---------:|-----------:|------------:|
| Random (analytic)                         |  0.5783 |     0.5000 | 0.2623 |   0.4289 |            |             |
| Popularity                                |  0.8166 |     0.6699 | 0.4350 |   0.0847 |     0.8166 |      0.0847 |
| Naive MF                                  |  0.8848 |     0.6979 | 0.4883 |   0.4385 |     0.9063 |      0.3397 |
| BPR-MF                                    |  0.9755 |     0.7622 | 0.5504 |   0.4371 |     0.9796 |      0.4339 |
| EASE                                      |  0.9134 |     0.6921 | 0.5130 |   0.4477 |     0.9715 |      0.4082 |
| WMF                                       |  1.0030 |     0.7801 | 0.5725 |   0.4719 |     1.0165 |      0.4521 |
| SCAR (constant θ)                         |  1.0049 |     0.7802 | 0.5745 |   0.4681 |     1.0052 |      0.4662 |
| Rel-MF                                    |  0.9732 |     0.7630 | 0.5515 |   0.4819 |     0.9716 |      0.4548 |
| Rel-MF + nnPU                             |  0.9587 |     0.7491 | 0.5416 |   0.4629 |     0.9775 |      0.4902 |
| Rel-MF balanced (official-code objective) |  0.9881 |     0.7647 | 0.5622 |   0.5105 |     1.0053 |      0.5294 |
| Ideal (analytic)                          |  1.5022 |     0.9932 | 1.0000 |   1.2594 |            |             |

### Main table, all cut-offs

|                                           |   DCG@1 |   DCG@3 |   DCG@5 |   Recall@1 |   Recall@3 |   Recall@5 |   AP@1 |   AP@3 |   AP@5 |   Rare@5 |   B: DCG@5 |   B: Rare@5 |
|:------------------------------------------|--------:|--------:|--------:|-----------:|-----------:|-----------:|-------:|-------:|-------:|---------:|-----------:|------------:|
| Random (analytic)                         |  0.1961 |  0.4179 |  0.5783 |     0.1000 |     0.3000 |     0.5000 | 0.1961 | 0.2149 | 0.2623 |   0.4289 |            |             |
| Popularity                                |  0.3887 |  0.6631 |  0.8166 |     0.2334 |     0.4867 |     0.6699 | 0.3887 | 0.3904 | 0.4350 |   0.0847 |     0.8166 |      0.0847 |
| Naive MF                                  |  0.4580 |  0.7430 |  0.8848 |     0.2834 |     0.5325 |     0.6979 | 0.4580 | 0.4493 | 0.4883 |   0.4385 |     0.9063 |      0.3397 |
| BPR-MF                                    |  0.5124 |  0.8293 |  0.9755 |     0.3182 |     0.5966 |     0.7622 | 0.5124 | 0.5099 | 0.5504 |   0.4371 |     0.9796 |      0.4339 |
| EASE                                      |  0.5007 |  0.7867 |  0.9134 |     0.3116 |     0.5556 |     0.6921 | 0.5007 | 0.4816 | 0.5130 |   0.4477 |     0.9715 |      0.4082 |
| WMF                                       |  0.5354 |  0.8620 |  1.0030 |     0.3345 |     0.6220 |     0.7801 | 0.5354 | 0.5337 | 0.5725 |   0.4719 |     1.0165 |      0.4521 |
| SCAR (constant θ)                         |  0.5345 |  0.8708 |  1.0049 |     0.3338 |     0.6327 |     0.7802 | 0.5345 | 0.5393 | 0.5745 |   0.4681 |     1.0052 |      0.4662 |
| Rel-MF                                    |  0.5177 |  0.8311 |  0.9732 |     0.3231 |     0.6031 |     0.7630 | 0.5177 | 0.5122 | 0.5515 |   0.4819 |     0.9716 |      0.4548 |
| Rel-MF + nnPU                             |  0.5098 |  0.8136 |  0.9587 |     0.3180 |     0.5876 |     0.7491 | 0.5098 | 0.5013 | 0.5416 |   0.4629 |     0.9775 |      0.4902 |
| Rel-MF balanced (official-code objective) |  0.5271 |  0.8509 |  0.9881 |     0.3293 |     0.6111 |     0.7647 | 0.5271 | 0.5246 | 0.5622 |   0.5105 |     1.0053 |      0.5294 |
| Ideal (analytic)                          |  1.0000 |  1.4285 |  1.5022 |     0.7020 |     0.9575 |     0.9932 | 1.0000 | 1.0000 | 1.0000 |   1.2594 |            |             |

### Paired differences (user-level bootstrap, 3 seeds)

| comparison                                          | rule   | metric            | diff    | 95% CI             | relative gain   | headroom share   | seeds agree   | reading             |
|:----------------------------------------------------|:-------|:------------------|:--------|:-------------------|:----------------|:-----------------|:--------------|:--------------------|
| Rel-MF vs Naive MF                                  | A      | DCG@5             | +0.0883 | [+0.0764, +0.1013] | +10.0%          | +14.3%           | yes           | higher              |
| Rel-MF vs Naive MF                                  | A      | Rare-credit DCG@5 | +0.0434 | [+0.0253, +0.0594] | +9.9%           | +5.3%            | yes           | higher              |
| Rel-MF vs Naive MF                                  | B      | DCG@5             | +0.0653 | [+0.0530, +0.0787] | +7.2%           | +11.0%           | yes           | higher              |
| Rel-MF vs Naive MF                                  | B      | Rare-credit DCG@5 | +0.1152 | [+0.0973, +0.1338] | +33.9%          | +12.5%           | yes           | higher              |
| Rel-MF + nnPU vs Rel-MF                             | A      | DCG@5             | -0.0145 | [-0.0237, -0.0045] | -1.5%           | -2.7%            | yes           | lower               |
| Rel-MF + nnPU vs Rel-MF                             | A      | Rare-credit DCG@5 | -0.0190 | [-0.0325, -0.0051] | -4.0%           | -2.4%            | yes           | lower               |
| Rel-MF + nnPU vs Rel-MF                             | B      | DCG@5             | +0.0059 | [-0.0007, +0.0122] | +0.6%           | +1.1%            | no            | no clear difference |
| Rel-MF + nnPU vs Rel-MF                             | B      | Rare-credit DCG@5 | +0.0354 | [+0.0266, +0.0443] | +7.8%           | +4.4%            | yes           | higher              |
| Rel-MF balanced (official-code objective) vs Rel-MF | A      | DCG@5             | +0.0150 | [+0.0026, +0.0280] | +1.5%           | +2.8%            | yes           | higher              |
| Rel-MF balanced (official-code objective) vs Rel-MF | A      | Rare-credit DCG@5 | +0.0286 | [+0.0115, +0.0448] | +5.9%           | +3.7%            | yes           | higher              |
| Rel-MF balanced (official-code objective) vs Rel-MF | B      | DCG@5             | +0.0337 | [+0.0206, +0.0465] | +3.5%           | +6.4%            | yes           | higher              |
| Rel-MF balanced (official-code objective) vs Rel-MF | B      | Rare-credit DCG@5 | +0.0745 | [+0.0575, +0.0920] | +16.4%          | +9.3%            | yes           | higher              |
| EASE vs Rel-MF                                      | A      | DCG@5             | -0.0597 | [-0.0753, -0.0448] | -6.1%           | -11.3%           | yes           | lower               |
| EASE vs Rel-MF                                      | A      | Rare-credit DCG@5 | -0.0343 | [-0.0550, -0.0132] | -7.1%           | -4.4%            | yes           | lower               |
| EASE vs Rel-MF                                      | B      | DCG@5             | -0.0001 | [-0.0140, +0.0132] | -0.0%           | -0.0%            | no            | no clear difference |
| EASE vs Rel-MF                                      | B      | Rare-credit DCG@5 | -0.0467 | [-0.0662, -0.0266] | -10.3%          | -5.8%            | yes           | lower               |
| BPR-MF vs Rel-MF                                    | A      | DCG@5             | +0.0024 | [-0.0084, +0.0140] | +0.2%           | +0.5%            | no            | no clear difference |
| BPR-MF vs Rel-MF                                    | A      | Rare-credit DCG@5 | -0.0449 | [-0.0591, -0.0297] | -9.3%           | -5.8%            | yes           | lower               |
| BPR-MF vs Rel-MF                                    | B      | DCG@5             | +0.0081 | [-0.0032, +0.0186] | +0.8%           | +1.5%            | no            | no clear difference |
| BPR-MF vs Rel-MF                                    | B      | Rare-credit DCG@5 | -0.0209 | [-0.0356, -0.0044] | -4.6%           | -2.6%            | yes           | lower               |

Rel-MF vs naive MF, rule A, DCG@5: held  
Rel-MF vs naive MF, rule A, Rare-credit DCG@5: held  
Rel-MF vs naive MF, rule B, DCG@5: held  
Rel-MF vs naive MF, rule B, Rare-credit DCG@5: held

### Rare items, two readings (rule A)

|                                           |   Credit DCG@1 |   Credit DCG@3 |   Credit DCG@5 |   Restricted DCG@1 |   Restricted DCG@3 |   Restricted Recall@1 |   Restricted Recall@3 |   Restricted Recall@5 (saturated) |
|:------------------------------------------|---------------:|---------------:|---------------:|-------------------:|-------------------:|----------------------:|----------------------:|----------------------------------:|
| Naive MF                                  |         0.0944 |         0.2789 |         0.4385 |             0.4362 |             0.7585 |                0.3259 |                0.7142 |                            0.9277 |
| WMF                                       |         0.1302 |         0.3310 |         0.4719 |             0.5266 |             0.8411 |                0.4088 |                0.7723 |                            0.9394 |
| SCAR (constant θ)                         |         0.1168 |         0.3262 |         0.4681 |             0.5266 |             0.8471 |                0.4077 |                0.7794 |                            0.9454 |
| Rel-MF                                    |         0.1597 |         0.3468 |         0.4819 |             0.4986 |             0.8077 |                0.3822 |                0.7432 |                            0.9342 |
| Rel-MF + nnPU                             |         0.1457 |         0.3283 |         0.4629 |             0.4762 |             0.7939 |                0.3650 |                0.7401 |                            0.9336 |
| Rel-MF balanced (official-code objective) |         0.1597 |         0.3755 |         0.5105 |             0.5039 |             0.8246 |                0.3858 |                0.7599 |                            0.9411 |
| Random (analytic)                         |         0.1455 |         0.3100 |         0.4289 |             0.2932 |             0.6110 |                0.2111 |                0.6078 |                            0.8914 |

Users with a rare relevant test item: 1073. Rare positives: 1561. Mean rare candidates per user: 4.99.

### Restricted reading, paired differences (3 seeds)

| comparison                                                     | rule   | metric           | diff    | 95% CI             | relative gain   | headroom share   | seeds agree   | reading             |
|:---------------------------------------------------------------|:-------|:-----------------|:--------|:-------------------|:----------------|:-----------------|:--------------|:--------------------|
| WMF vs Rel-MF balanced (official-code objective)               | A      | Restricted DCG@1 | +0.0227 | [+0.0068, +0.0410] | +4.5%           | +4.6%            | yes           | higher              |
| SCAR (constant θ) vs Rel-MF balanced (official-code objective) | A      | Restricted DCG@1 | +0.0227 | [+0.0009, +0.0460] | +4.5%           | +4.6%            | yes           | higher              |
| WMF vs Rel-MF                                                  | A      | Restricted DCG@1 | +0.0280 | [+0.0043, +0.0522] | +5.6%           | +5.6%            | yes           | higher              |
| SCAR (constant θ) vs Rel-MF                                    | A      | Restricted DCG@1 | +0.0280 | [+0.0043, +0.0516] | +5.6%           | +5.6%            | yes           | higher              |
| WMF vs Rel-MF balanced (official-code objective)               | A      | Restricted DCG@3 | +0.0165 | [+0.0054, +0.0283] | +2.0%           | +4.0%            | yes           | higher              |
| SCAR (constant θ) vs Rel-MF balanced (official-code objective) | A      | Restricted DCG@3 | +0.0226 | [+0.0076, +0.0388] | +2.7%           | +5.5%            | yes           | higher              |
| WMF vs Rel-MF                                                  | A      | Restricted DCG@3 | +0.0334 | [+0.0174, +0.0516] | +4.1%           | +7.8%            | yes           | higher              |
| SCAR (constant θ) vs Rel-MF                                    | A      | Restricted DCG@3 | +0.0394 | [+0.0216, +0.0574] | +4.9%           | +9.2%            | yes           | higher              |
| WMF vs Rel-MF balanced (official-code objective)               | B      | Restricted DCG@1 | -0.0096 | [-0.0267, +0.0093] | -1.8%           | -2.1%            | yes           | no clear difference |
| SCAR (constant θ) vs Rel-MF balanced (official-code objective) | B      | Restricted DCG@1 | -0.0162 | [-0.0413, +0.0093] | -3.0%           | -3.5%            | yes           | no clear difference |
| WMF vs Rel-MF                                                  | B      | Restricted DCG@1 | +0.0270 | [+0.0037, +0.0497] | +5.4%           | +5.4%            | yes           | higher              |
| SCAR (constant θ) vs Rel-MF                                    | B      | Restricted DCG@1 | +0.0205 | [-0.0022, +0.0444] | +4.1%           | +4.1%            | yes           | no clear difference |
| WMF vs Rel-MF balanced (official-code objective)               | B      | Restricted DCG@3 | +0.0009 | [-0.0100, +0.0124] | +0.1%           | +0.2%            | no            | no clear difference |
| SCAR (constant θ) vs Rel-MF balanced (official-code objective) | B      | Restricted DCG@3 | -0.0036 | [-0.0208, +0.0134] | -0.4%           | -0.9%            | yes           | no clear difference |
| WMF vs Rel-MF                                                  | B      | Restricted DCG@3 | +0.0312 | [+0.0134, +0.0492] | +3.8%           | +7.5%            | yes           | higher              |
| SCAR (constant θ) vs Rel-MF                                    | B      | Restricted DCG@3 | +0.0266 | [+0.0093, +0.0437] | +3.3%           | +6.4%            | yes           | higher              |

Restricted DCG@1 under rule B: WMF 0.5294, SCAR (constant θ) 0.5228, Rel-MF 0.5023, Rel-MF balanced (official-code objective) 0.5390.
