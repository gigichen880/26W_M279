# Collapse decomposition

Train fit: 2008–2012. Query predict: 2013–2016. No 2017+ used.

Memberships are from `cmeans_predict` for both train and query (same path as walk-forward queries).

```
                    tag  dim  geom_train_cv  geom_query_cv  geom_train_rel_contrast  geom_query_rel_contrast  fcm_train_mean_max_p  fcm_query_mean_max_p  fcm_train_mean_norm_entropy  fcm_query_mean_norm_entropy  cent_train_mean_centroid_rel_contrast  cent_query_mean_centroid_rel_contrast
      pca_svd_legacy_48  240       0.478139       1.452662                10.838125                51.334091              0.250000              0.250000                     1.000000                     1.000000                           5.428524e-08                           5.240190e-08
            pca_only_48   48       0.462937       1.434070                10.419547                53.569386              0.250000              0.250000                     1.000000                     1.000000                           5.594415e-08                           5.205307e-08
pca_svd_standardized_48  240       0.483414       0.997522                12.805972                33.159726              0.531014              0.524970                     0.799332                     0.797286                           1.966755e+00                           2.244738e+00
            pca_only_15   15       0.587830       1.537487                20.856592                86.024180              0.250000              0.250000                     1.000000                     1.000000                           7.488452e-07                           8.356743e-07
            pca_only_10   10       0.642227       1.599681                38.415730               132.819123              0.250000              0.250000                     1.000000                     1.000000                           1.588932e-06                           1.820247e-06
             pca_only_5    5       0.755536       1.676503               133.671880               428.191662              0.250680              0.250879                     0.999998                     0.999995                           2.759256e-03                           3.549266e-03
         market_state_6    6       0.510361       0.647068               833.258933               189.767642              0.670492              0.639727                     0.609923                     0.628448                           3.981277e+00                           3.938781e+00
```
