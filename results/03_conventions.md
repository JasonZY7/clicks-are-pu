### One pair of models, six conventions (rule A, 3 seeds)

| convention                                                          |   DCG@5 naive |   DCG@5 Rel-MF |   DCG gain |   Recall@5 naive |   Recall@5 Rel-MF |   Recall gain |   AP@5 naive |   AP@5 Rel-MF |   AP gain |
|:--------------------------------------------------------------------|--------------:|---------------:|-----------:|-----------------:|------------------:|--------------:|-------------:|--------------:|----------:|
| This project                                                        |        0.8848 |         0.9732 |     0.0998 |           0.6979 |            0.7630 |        0.0933 |       0.4883 |        0.5515 |    0.1293 |
| Official discount and unnormalised AP                               |        1.0361 |         1.1380 |     0.0983 |           0.6979 |            0.7630 |        0.0933 |       0.9423 |        1.0589 |    0.1238 |
| Zero-positive users in the mean                                     |        0.4000 |         0.4399 |     0.0998 |           0.5895 |            0.6189 |        0.0499 |       0.2208 |        0.2493 |    0.1293 |
| Rare restricted, bottom 50% of items                                |        0.8963 |         0.9340 |     0.0421 |           0.9277 |            0.9342 |        0.0070 |       0.5980 |        0.6407 |    0.0715 |
| Rare restricted, n_i ≤ 500 (96% of items)                           |        0.8422 |         0.9289 |     0.1030 |           0.6959 |            0.7555 |        0.0857 |       0.4620 |        0.5270 |    0.1406 |
| 25cfa65 evaluator, re-implemented (all users, relevance floor 0.01) |        0.4981 |         0.5438 |     0.0918 |           0.5848 |            0.6121 |        0.0466 |       0.4256 |        0.4789 |    0.1253 |

This project: the paper's DCG discount, standard AP, users with at least one relevant test item. Official discount: ranks 1 and 2 weigh the same. Zero-positive users count as 0 for DCG and AP and get the official Recall@5 of 5 / candidates. Gains are relative, Rel-MF over naive MF.

### DCG@5 on a re-implementation of the 25cfa65 evaluator (rule A, 3 seeds; read-offs are from arXiv v3, p. 8, accurate to about 0.0005)

|                                            |   DCG@5 |
|:-------------------------------------------|--------:|
| Random (analytic)                          |  0.3457 |
| Popularity                                 |  0.4669 |
| Naive MF                                   |  0.4981 |
| EASE                                       |  0.5121 |
| BPR-MF                                     |  0.5467 |
| Rel-MF                                     |  0.5438 |
| Rel-MF balanced                            |  0.5534 |
| WMF                                        |  0.5611 |
| SCAR                                       |  0.5634 |
| Paper, printed Figure 2, WMF (read off)    |  0.3640 |
| Paper, printed Figure 2, ExpoMF (read off) |  0.4030 |
| Paper, printed Figure 2, Rel-MF (read off) |  0.4850 |
