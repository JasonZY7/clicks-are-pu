### Share of ratings by star

|                             |   1 stars |   2 stars |   3 stars |   4 stars |   5 stars |
|:----------------------------|----------:|----------:|----------:|----------:|----------:|
| Yahoo! R3, self-selected    |     0.314 |     0.127 |     0.158 |     0.156 |     0.246 |
| Yahoo! R3, uniform exposure |     0.526 |     0.242 |     0.144 |     0.062 |     0.026 |
| Coat, self-selected         |     0.273 |     0.206 |     0.247 |     0.183 |     0.091 |
| Coat, uniform exposure      |     0.405 |     0.194 |     0.216 |     0.138 |     0.047 |

### Datasets and splits

|                                                                | Yahoo! R3                  | Coat                   |
|:---------------------------------------------------------------|:---------------------------|:-----------------------|
| users                                                          | 15400                      | 290                    |
| items                                                          | 1000                       | 300                    |
| self-selected ratings                                          | 311704                     | 6960                   |
| ratings per user, min to max                                   | 10 to 648                  | 24 to 24               |
| self-selected share with rating >= 4                           | 0.401                      | 0.274                  |
| uniform-exposure ratings                                       | 54000                      | 4640                   |
| uniform-exposure share with rating >= 4                        | 0.088                      | 0.185                  |
| training positives (fit)                                       | 112570                     | 1715                   |
| held-out positives (val_A)                                     | 12507                      | 190                    |
| training label density                                         | 0.7310%                    | 1.9713%                |
| label frequency c* on val_B                                    | 0.090                      | 0.118                  |
| val_B users / rows / positives / users with a positive         | 540 / 5400 / 440 / 239     | 29 / 426 / 71 / 25     |
| test users / rows / positives / users with a positive          | 4860 / 48600 / 4309 / 2197 | 261 / 3848 / 698 / 200 |
| uniform-exposure cells also rated in training (removed if any) | 0 (0 positive)             | 366 (91 positive)      |

### Popularity concentration

|                                                                   | Yahoo! R3                                | Coat                                   |
|:------------------------------------------------------------------|:-----------------------------------------|:---------------------------------------|
| top 10% of items, share of positives                              | 53.7%                                    | 33.2%                                  |
| most-rated item, share of users who rated it                      | 36.0%                                    | 30.3%                                  |
| hottest item, share of users who rated it                         | 34.2%                                    | 28.6%                                  |
| hottest item, click rate / test relevance rate = implied exposure | 0.149 / 0.39 = 0.38 (18 of 46 test rows) | 0.169 / 0.43 = 0.39 (3 of 7 test rows) |
| items with no training positive                                   | 0                                        | 21                                     |
| rare items: count, largest n_i                                    | 500, 47                                  | 150, 4                                 |
| Spearman(n_i, ratings per item)                                   | 0.85                                     | 0.67                                   |

### Exposure weights on training positives, by η

| dataset   | η    | min non-zero θ̂   | mean weight   | max weight   |
|:----------|:-----|:-----------------|:--------------|:-------------|
| Yahoo! R3 | 0.25 | 0.1902           | 1.82          | 5.3          |
| Yahoo! R3 | 0.5  | 0.0362           | 3.70          | 27.7         |
| Yahoo! R3 | 1    | 0.0013           | 20.38         | 764.7        |
| Coat      | 0.25 | 0.3780           | 1.58          | 2.6          |
| Coat      | 0.5  | 0.1429           | 2.59          | 7.0          |
| Coat      | 1    | 0.0204           | 7.97          | 49.0         |
