## GNSS derived Precipitable Water Vapor (PWV) analysis and ML tools
This repository is used mainly to extract and analyze the PWV that is derived from the Global Navigation Satellite System (GNSS) radio signals received at ground stations.

Three major projects have come to fruition from using this repository (so far:-):

### 1) Binary classification on GPS derived PWV data for predicting flash floods in the Eastern Mediteranean
#### Project description
In the Eastern Mediteranean (EM) area (e.g., Israel), flash floods are mainly the result of heavy rainfall events which in turn can be detected by the rise of the water vapor content in the air. GPS ground based receivers can constantely monitor the water vapor amount in the air above the receiver, thus potentially become a flash floods warning system. Here we use the Precipitable Water Vapor (PWV) derived from 9 GPS ground based stations in the arid part of the EM region in order to predict flash floods. Our approach includes using three types of Machine Learning (ML) models in a binary classification task which predicts whether or not a flash flood will occurgiven 24 hours of PWV data. We train our models on 107 unique flash flood events and vigorously test them using a nested crossvalidation  technique. The models are further improved by adding more features (e.g., surface pressure measurements).
This project is yet to be published, however this is a tentative title:
>Flash floods prediction using precipitable water vapor derived from tropospheric path delays over the Eastern Mediterranean - to be published soon"



#### Example for multi-scorer, multi-model testing score visualization with nested cross validation as uncertainty
![](Figures/ML_scores_models_nested_CV_doy_pressure_pwv_pwv+pressure_pwv+pressure+doy_5_neg_1.png?raw=true "RF")
Mean test scores for the SVM, RF and MLP classifiers (row) and for each scorer (column). The feature groups consist of day of year (purple), surface pressure (brown), PWV (blue), PWV and surface pressure (orange) and all three together (green). The mean scores are indicated to the upper left of each bar and the standard deviation of 5 data splits is represented by the error bar length.

#### Example for Hyper-parameter tuning visualization using grid search and 2 schemes of data splits
![](Figures/Hyper-parameters_nested_RF.png?raw=true "RF")
A panel showing the optimal hyper parameters that were found using grid search cross validation for the RF classifier. Each set of hyper parameters were found for each outer split (row) and for each scorer (column). Moreover, the hyper parameters are also optimized when considering 4 and 5 inner folds(top and bottom panels  respectively). Each hyper parameter name is denoted in the title of each top panel and its values are indicated in the accompanied colorbar. 

### 2) Diurnal climatology analysis on GPS derived PWV data to predict flash floods in the Eastern Mediteranean:

#### Project description
In this project we take the long term PWV times series with a 5-min resolution and study its diurnal cycle using various tools (e.g., harmonic analysis).
You can read about it in the paper published:
>Ziv, S. Z., Yair, Y., Alpert, P., Uzan, L., & Reuveni, Y. (2021). The diurnal variability of precipitable water vapor derived from GPS tropospheric path delays over the Eastern Mediterranean. Atmospheric Research, 249, 105307.

BibTex entry:
```
@article{ziv2021diurnal,
  title={The diurnal variability of precipitable water vapor derived from GPS tropospheric path delays over the Eastern Mediterranean},
  author={Ziv, Shlomi Ziskin and Yair, Yoav and Alpert, Pinhas and Uzan, Leenes and Reuveni, Yuval},
  journal={Atmospheric Research},
  volume={249},
  pages={105307},
  year={2021},
  publisher={Elsevier}
}
```

### 3) Long-term climatology analysis on GPS derived PWV in the Eastern Mediteranean:

#### Project description
In this project we take the long term PWV times series, averaged to monthly means, and study their long term trends and inter-annual variations.
You can read about it in the paper published:
>S.  Ziskin  Ziv,  P.  Alpert,  and  Y.  Reuveni,  “Long  term  variability  and trends of precipitable water vapor derived from GPS tropospheric path delays over the Eastern Mediterranean, International Journal of Climatology.

BiBTex entry:
```
@article{ziskinlong,
  title={Long term variability and trends of precipitable water vapor derived from GPS tropospheric path delays over the Eastern Mediterranean},
  author={Ziskin Ziv, Shlomi and Alpert, Pinhas and Reuveni, Yuval},
  journal={International Journal of Climatology},
  publisher={Wiley Online Library}
}
```
## License
The code is free for use under the MIT license and if it helps you in an academic publication, i'd be happy if you cite the relevant aforementioned works in your paper.

## Authors
* **Shlomi Ziskin Ziv** - *shlomiziskin@gmail.com*

