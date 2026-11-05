### WMF(c′) and SCAR(1/c′), with the Rel-MF and balanced winners below: test DCG@5 / rare-credit DCG@5 (seed 42; c′ = 500 is the control)

| c′ (WMF) = 1/c (SCAR)   | WMF, rule A     | SCAR, rule A    | WMF, rule B     | SCAR, rule B    |
|:------------------------|:----------------|:----------------|:----------------|:----------------|
| 3                       | 0.9303 / 0.2620 | 0.9288 / 0.3575 | 0.9241 / 0.3157 | 0.9341 / 0.3511 |
| 10                      | 0.9744 / 0.3259 | 0.9535 / 0.4298 | 0.9736 / 0.3254 | 0.9768 / 0.4009 |
| 30                      | 0.9882 / 0.5132 | 1.0049 / 0.4683 | 0.9980 / 0.4127 | 1.0045 / 0.4658 |
| 100                     | 1.0051 / 0.4698 | 0.9846 / 0.4962 | 1.0063 / 0.4777 | 1.0071 / 0.4310 |
| 300                     | 0.9795 / 0.4737 | not in grid     | 1.0175 / 0.4536 | not in grid     |
| 500                     | not in grid     | 0.9251 / 0.4931 | not in grid     | 1.0014 / 0.4463 |
| 1000                    | 1.0139 / 0.4759 | not in grid     | 1.0153 / 0.4766 | not in grid     |
| Rel-MF                  | 0.9729 / 0.4935 |                 | 0.9746 / 0.4673 |                 |
| Rel-MF balanced         | 0.9869 / 0.5081 |                 | 1.0053 / 0.5298 |                 |

### Family winners (3 seeds)

| comparison                                                     | rule   | metric            | diff    | 95% CI             | relative gain   | headroom share   | seeds agree   | reading             |
|:---------------------------------------------------------------|:-------|:------------------|:--------|:-------------------|:----------------|:-----------------|:--------------|:--------------------|
| WMF vs Naive MF                                                | A      | DCG@5             | +0.1182 | [+0.1050, +0.1318] | +13.4%          | +19.1%           | yes           | higher              |
| WMF vs Naive MF                                                | A      | Rare-credit DCG@5 | +0.0334 | [+0.0151, +0.0508] | +7.6%           | +4.1%            | yes           | higher              |
| WMF vs Naive MF                                                | B      | DCG@5             | +0.1102 | [+0.0960, +0.1241] | +12.2%          | +18.5%           | yes           | higher              |
| WMF vs Naive MF                                                | B      | Rare-credit DCG@5 | +0.1125 | [+0.0923, +0.1332] | +33.1%          | +12.2%           | yes           | higher              |
| SCAR (constant θ) vs WMF                                       | A      | DCG@5             | +0.0019 | [-0.0037, +0.0081] | +0.2%           | +0.4%            | no            | no clear difference |
| SCAR (constant θ) vs WMF                                       | A      | Rare-credit DCG@5 | -0.0038 | [-0.0119, +0.0046] | -0.8%           | -0.5%            | yes           | no clear difference |
| SCAR (constant θ) vs WMF                                       | B      | DCG@5             | -0.0113 | [-0.0202, -0.0015] | -1.1%           | -2.3%            | yes           | lower               |
| SCAR (constant θ) vs WMF                                       | B      | Rare-credit DCG@5 | +0.0140 | [+0.0006, +0.0283] | +3.1%           | +1.7%            | yes           | higher              |
| Rel-MF vs SCAR (constant θ)                                    | A      | DCG@5             | -0.0318 | [-0.0432, -0.0206] | -3.2%           | -6.4%            | yes           | lower               |
| Rel-MF vs SCAR (constant θ)                                    | A      | Rare-credit DCG@5 | +0.0138 | [-0.0027, +0.0290] | +2.9%           | +1.7%            | yes           | no clear difference |
| Rel-MF vs SCAR (constant θ)                                    | B      | DCG@5             | -0.0336 | [-0.0454, -0.0230] | -3.3%           | -6.8%            | yes           | lower               |
| Rel-MF vs SCAR (constant θ)                                    | B      | Rare-credit DCG@5 | -0.0114 | [-0.0267, +0.0034] | -2.4%           | -1.4%            | no            | no clear difference |
| Rel-MF vs WMF                                                  | A      | DCG@5             | -0.0299 | [-0.0412, -0.0191] | -3.0%           | -6.0%            | yes           | lower               |
| Rel-MF vs WMF                                                  | A      | Rare-credit DCG@5 | +0.0100 | [-0.0062, +0.0262] | +2.1%           | +1.3%            | yes           | no clear difference |
| Rel-MF vs WMF                                                  | B      | DCG@5             | -0.0449 | [-0.0556, -0.0338] | -4.4%           | -9.2%            | yes           | lower               |
| Rel-MF vs WMF                                                  | B      | Rare-credit DCG@5 | +0.0027 | [-0.0151, +0.0194] | +0.6%           | +0.3%            | no            | no clear difference |
| Rel-MF balanced (official-code objective) vs WMF               | A      | DCG@5             | -0.0149 | [-0.0227, -0.0073] | -1.5%           | -3.0%            | yes           | lower               |
| Rel-MF balanced (official-code objective) vs WMF               | A      | Rare-credit DCG@5 | +0.0386 | [+0.0277, +0.0492] | +8.2%           | +4.9%            | yes           | higher              |
| Rel-MF balanced (official-code objective) vs WMF               | B      | DCG@5             | -0.0112 | [-0.0206, -0.0022] | -1.1%           | -2.3%            | yes           | lower               |
| Rel-MF balanced (official-code objective) vs WMF               | B      | Rare-credit DCG@5 | +0.0772 | [+0.0656, +0.0898] | +17.1%          | +9.6%            | yes           | higher              |
| Rel-MF balanced (official-code objective) vs SCAR (constant θ) | A      | DCG@5             | -0.0168 | [-0.0265, -0.0070] | -1.7%           | -3.4%            | yes           | lower               |
| Rel-MF balanced (official-code objective) vs SCAR (constant θ) | A      | Rare-credit DCG@5 | +0.0424 | [+0.0291, +0.0562] | +9.1%           | +5.4%            | yes           | higher              |
| Rel-MF balanced (official-code objective) vs SCAR (constant θ) | B      | DCG@5             | +0.0001 | [-0.0126, +0.0128] | +0.0%           | +0.0%            | no            | no clear difference |
| Rel-MF balanced (official-code objective) vs SCAR (constant θ) | B      | Rare-credit DCG@5 | +0.0632 | [+0.0463, +0.0801] | +13.6%          | +8.0%            | yes           | higher              |
| Rel-MF balanced (official-code objective) vs SCAR, c′ = 500    | A      | DCG@5             | +0.0628 | [+0.0530, +0.0731] | +6.8%           | +10.9%           | yes           | higher              |
| Rel-MF balanced (official-code objective) vs SCAR, c′ = 500    | A      | Rare-credit DCG@5 | +0.0217 | [+0.0087, +0.0337] | +4.4%           | +2.8%            | yes           | higher              |
| Rel-MF balanced (official-code objective) vs SCAR, c′ = 500    | B      | DCG@5             | +0.0036 | [-0.0043, +0.0118] | +0.4%           | +0.7%            | yes           | no clear difference |
| Rel-MF balanced (official-code objective) vs SCAR, c′ = 500    | B      | Rare-credit DCG@5 | +0.0802 | [+0.0691, +0.0911] | +17.8%          | +9.9%            | yes           | higher              |

### CI signs

- Rule A, DCG@5: SCAR ≈ WMF; Rel-MF < SCAR
- Rule A, Rare-credit DCG@5: SCAR ≈ WMF; Rel-MF ≈ SCAR
- Rule B, DCG@5: SCAR < WMF; Rel-MF < SCAR
- Rule B, Rare-credit DCG@5: SCAR > WMF; Rel-MF ≈ SCAR

### The PU term alone

Rule B, 3 seeds, test DCG@5 / rare-credit, no CI: SCAR c′ = 30 minus WMF c′ = 100: +0.0016 / -0.0090. WMF c′ = 300 minus WMF c′ = 100: +0.0130 / -0.0230.  
Same c′ and λ̃, cells of 16 where SCAR is above WMF (seed 42): rule A all items 8, rare-credit 13; rule B all items 12, rare-credit 15.  
Grid points of other families above balanced's rule A pick on both metrics (seed 42): wmf c′=30 λ̃=1 step 310, wmf c′=30 λ̃=1 step 320, wmf c′=30 λ̃=1 step 330  
Control at balanced's rule A λ̃ = 0.3, rule A inside that cell (seed 42): 0.9876 / 0.4896, step 2970.

### Popularity penalty at ranking time, WMF rule B against the row named in `against` (3 seeds, no training)

| α from                                                         | against   | metric            |    diff |     low |    high |
|:---------------------------------------------------------------|:----------|:------------------|--------:|--------:|--------:|
| top of val_B (α = [0.1, 0.18, 0.18])                           | WMF       | All items DCG@1   | +0.0006 | -0.0055 | +0.0070 |
| top of val_B (α = [0.1, 0.18, 0.18])                           | WMF       | Rare-credit DCG@1 | +0.0261 | +0.0180 | +0.0339 |
| top of val_B (α = [0.1, 0.18, 0.18])                           | WMF       | All items DCG@3   | -0.0006 | -0.0060 | +0.0053 |
| top of val_B (α = [0.1, 0.18, 0.18])                           | WMF       | Rare-credit DCG@3 | +0.0495 | +0.0413 | +0.0570 |
| top of val_B (α = [0.1, 0.18, 0.18])                           | WMF       | All items DCG@5   | -0.0047 | -0.0100 | +0.0007 |
| top of val_B (α = [0.1, 0.18, 0.18])                           | WMF       | Rare-credit DCG@5 | +0.0515 | +0.0445 | +0.0584 |
| balanced's val_B rare-credit at DCG@5 (α = [0.26, 0.24, 0.22]) | balanced  | All items DCG@1   | +0.0085 | -0.0035 | +0.0209 |
| balanced's val_B rare-credit at DCG@5 (α = [0.26, 0.24, 0.22]) | balanced  | Rare-credit DCG@1 | -0.0205 | -0.0342 | -0.0068 |
| balanced's val_B rare-credit at DCG@5 (α = [0.26, 0.24, 0.22]) | balanced  | All items DCG@3   | +0.0072 | -0.0019 | +0.0171 |
| balanced's val_B rare-credit at DCG@5 (α = [0.26, 0.24, 0.22]) | balanced  | Rare-credit DCG@3 | -0.0115 | -0.0223 | -0.0006 |
| balanced's val_B rare-credit at DCG@5 (α = [0.26, 0.24, 0.22]) | balanced  | All items DCG@5   | -0.0006 | -0.0086 | +0.0079 |
| balanced's val_B rare-credit at DCG@5 (α = [0.26, 0.24, 0.22]) | balanced  | Rare-credit DCG@5 | +0.0046 | -0.0060 | +0.0157 |
| balanced's val_B rare-credit at DCG@3 (α = [0.2, 0.2, 0.22])   | balanced  | All items DCG@1   | +0.0103 | -0.0018 | +0.0234 |
| balanced's val_B rare-credit at DCG@3 (α = [0.2, 0.2, 0.22])   | balanced  | Rare-credit DCG@1 | -0.0280 | -0.0416 | -0.0146 |
| balanced's val_B rare-credit at DCG@3 (α = [0.2, 0.2, 0.22])   | balanced  | All items DCG@3   | +0.0109 | +0.0014 | +0.0208 |
| balanced's val_B rare-credit at DCG@3 (α = [0.2, 0.2, 0.22])   | balanced  | Rare-credit DCG@3 | -0.0246 | -0.0355 | -0.0134 |
| balanced's val_B rare-credit at DCG@3 (α = [0.2, 0.2, 0.22])   | balanced  | All items DCG@5   | +0.0030 | -0.0046 | +0.0110 |
| balanced's val_B rare-credit at DCG@3 (α = [0.2, 0.2, 0.22])   | balanced  | Rare-credit DCG@5 | -0.0081 | -0.0182 | +0.0029 |
| balanced's val_B rare-credit at DCG@1 (α = [0.28, 0.28, 0.34]) | balanced  | All items DCG@1   | +0.0080 | -0.0033 | +0.0196 |
| balanced's val_B rare-credit at DCG@1 (α = [0.28, 0.28, 0.34]) | balanced  | Rare-credit DCG@1 | -0.0087 | -0.0217 | +0.0040 |
| balanced's val_B rare-credit at DCG@1 (α = [0.28, 0.28, 0.34]) | balanced  | All items DCG@3   | -0.0013 | -0.0110 | +0.0087 |
| balanced's val_B rare-credit at DCG@1 (α = [0.28, 0.28, 0.34]) | balanced  | Rare-credit DCG@3 | +0.0110 | -0.0010 | +0.0226 |
| balanced's val_B rare-credit at DCG@1 (α = [0.28, 0.28, 0.34]) | balanced  | All items DCG@5   | -0.0072 | -0.0155 | +0.0013 |
| balanced's val_B rare-credit at DCG@1 (α = [0.28, 0.28, 0.34]) | balanced  | Rare-credit DCG@5 | +0.0265 | +0.0153 | +0.0379 |

### All comparisons, difference and 95% CI (3 seeds; bold when the CI excludes 0)

| comparison                        | All items, rule A              | Rare-credit, rule A            | All items, rule B              | Rare-credit, rule B            |
|:----------------------------------|:-------------------------------|:-------------------------------|:-------------------------------|:-------------------------------|
| Rel-MF vs Naive MF                | **+0.0883** [+0.0764, +0.1013] | **+0.0434** [+0.0253, +0.0594] | **+0.0653** [+0.0530, +0.0787] | **+0.1152** [+0.0973, +0.1338] |
| WMF vs Naive MF                   | **+0.1182** [+0.1050, +0.1318] | **+0.0334** [+0.0151, +0.0508] | **+0.1102** [+0.0960, +0.1241] | **+0.1125** [+0.0923, +0.1332] |
| SCAR vs WMF                       | +0.0019 [-0.0037, +0.0081]     | -0.0038 [-0.0119, +0.0046]     | **-0.0113** [-0.0202, -0.0015] | **+0.0140** [+0.0006, +0.0283] |
| Rel-MF vs SCAR                    | **-0.0318** [-0.0432, -0.0206] | +0.0138 [-0.0027, +0.0290]     | **-0.0336** [-0.0454, -0.0230] | -0.0114 [-0.0267, +0.0034]     |
| Rel-MF vs WMF                     | **-0.0299** [-0.0412, -0.0191] | +0.0100 [-0.0062, +0.0262]     | **-0.0449** [-0.0556, -0.0338] | +0.0027 [-0.0151, +0.0194]     |
| Rel-MF balanced vs WMF            | **-0.0149** [-0.0227, -0.0073] | **+0.0386** [+0.0277, +0.0492] | **-0.0112** [-0.0206, -0.0022] | **+0.0772** [+0.0656, +0.0898] |
| Rel-MF balanced vs SCAR           | **-0.0168** [-0.0265, -0.0070] | **+0.0424** [+0.0291, +0.0562] | +0.0001 [-0.0126, +0.0128]     | **+0.0632** [+0.0463, +0.0801] |
| Rel-MF balanced vs SCAR, c′ = 500 | **+0.0628** [+0.0530, +0.0731] | **+0.0217** [+0.0087, +0.0337] | +0.0036 [-0.0043, +0.0118]     | **+0.0802** [+0.0691, +0.0911] |
| Rel-MF + nnPU vs Rel-MF           | **-0.0145** [-0.0237, -0.0045] | **-0.0190** [-0.0325, -0.0051] | +0.0059 [-0.0007, +0.0122]     | **+0.0354** [+0.0266, +0.0443] |
| Rel-MF balanced vs Rel-MF         | **+0.0150** [+0.0026, +0.0280] | **+0.0286** [+0.0115, +0.0448] | **+0.0337** [+0.0206, +0.0465] | **+0.0745** [+0.0575, +0.0920] |
| EASE vs Rel-MF                    | **-0.0597** [-0.0753, -0.0448] | **-0.0343** [-0.0550, -0.0132] | -0.0001 [-0.0140, +0.0132]     | **-0.0467** [-0.0662, -0.0266] |
| BPR-MF vs Rel-MF                  | +0.0024 [-0.0084, +0.0140]     | **-0.0449** [-0.0591, -0.0297] | +0.0081 [-0.0032, +0.0186]     | **-0.0209** [-0.0356, -0.0044] |

### Each family's winner at seed 42 (finals re-select the step per seed; BPR-MF's step is an epoch, its λ̃ the regulariser; EASE's λ̃ is λ)

| family           | rule   |   c′ (WMF) = 1/c (SCAR) |    η |   M |       λ̃ |   step |   test DCG@5 |   rare-credit |
|:-----------------|:-------|------------------------:|-----:|----:|--------:|-------:|-------------:|--------------:|
| Popularity       | A      |                         |      |     |         |      0 |       0.8166 |        0.0847 |
| Popularity       | B      |                         |      |     |         |      0 |       0.8166 |        0.0847 |
| Naive MF         | A      |                         |      |     |    1    |   4000 |       0.8944 |        0.4137 |
| Naive MF         | B      |                         |      |     |    1    |   2300 |       0.9059 |        0.3287 |
| BPR-MF           | A      |                         |      |     |    0.01 |     57 |       0.9712 |        0.4379 |
| BPR-MF           | B      |                         |      |     |    0.01 |     68 |       0.9713 |        0.4320 |
| EASE             | A      |                         |      |     |  100    |      0 |       0.9134 |        0.4477 |
| EASE             | B      |                         |      |     | 1000    |      0 |       0.9715 |        0.4082 |
| WMF              | A      |                     100 |      |     |    1    |   1170 |       1.0051 |        0.4698 |
| WMF              | B      |                     300 |      |     |    1    |    440 |       1.0175 |        0.4536 |
| SCAR             | A      |                      30 |      |     |    3    |   3000 |       1.0049 |        0.4683 |
| SCAR             | B      |                      30 |      |     |    3    |   1820 |       1.0045 |        0.4658 |
| Rel-MF           | A      |                         | 0.5  | 0.1 |    3    |    980 |       0.9729 |        0.4935 |
| Rel-MF           | B      |                         | 0.5  | 0.1 |    0.3  |    450 |       0.9746 |        0.4673 |
| Rel-MF + nnPU    | A      |                         | 0.5  | 0   |    3    |   2140 |       0.9592 |        0.4644 |
| Rel-MF + nnPU    | B      |                         | 0.5  | 0   |    0.3  |    550 |       0.9842 |        0.4929 |
| Rel-MF balanced  | A      |                         | 0.5  | 0   |    0.3  |   2970 |       0.9869 |        0.5081 |
| Rel-MF balanced  | B      |                         | 0.5  | 0   |    1    |   2610 |       1.0053 |        0.5298 |
| Rel-MF, η = 0.25 | A      |                         | 0.25 | 0   |    3    |   2740 |       0.9307 |        0.4036 |
| Rel-MF, η = 1    | A      |                         | 1    | 0.1 |    3    |    730 |       0.9508 |        0.4759 |
