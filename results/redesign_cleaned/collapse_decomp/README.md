# Collapse decomposition

Train fit: 2008–2012. Query predict: 2013–2016. No 2017+ used.

Memberships are from `cmeans_predict` for both train and query (same path as walk-forward queries).

```
                    tag  dim  geom_train_cv  geom_query_cv  geom_train_rel_contrast  geom_query_rel_contrast  fcm_train_mean_max_p  fcm_query_mean_max_p  fcm_train_mean_norm_entropy  fcm_query_mean_norm_entropy  cent_train_mean_centroid_rel_contrast  cent_query_mean_centroid_rel_contrast
      pca_svd_legacy_48  240       0.489121       0.205086                 9.788712                 2.436449              0.250000              0.250000                     1.000000                     1.000000                           2.269123e-07                           2.488011e-07
            pca_only_48   48       0.464633       0.211200                 9.306364                 2.604642              0.250000              0.250000                     1.000000                     1.000000                           2.346356e-07                           2.629985e-07
pca_svd_standardized_48  240       0.432504       0.314483                 6.736805                 5.540396              0.438836              0.421291                     0.881975                     0.881012                           1.443886e+00                           1.524086e+00
            pca_only_15   15       0.610266       0.228744                22.524494                 5.112810              0.250000              0.250000                     1.000000                     1.000000                           5.190477e-07                           6.448621e-07
            pca_only_10   10       0.646621       0.246643                38.957665                 7.628528              0.250000              0.250000                     1.000000                     1.000000                           1.328834e-06                           1.473552e-06
             pca_only_5    5       0.788846       0.346246               146.795361                30.827580              0.275429              0.281138                     0.993934                     0.991209                           1.143828e-01                           1.452959e-01
         market_state_6    6       0.434227       0.430121               257.275923                78.801533              0.591904              0.522437                     0.736056                     0.792635                           2.220390e+00                           1.951099e+00
```
