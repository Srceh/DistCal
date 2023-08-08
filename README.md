# DistCal
Code for the ICML 2019 paper: Distribution Calibration for Regression

This implementation uses Tensorflow 2.0 (https://www.tensorflow.org/) as backend for automatic differentiation and GPU acceleration.

example.ipynb contains the example code of applying the GP-Beta approach on the Boston dataset and ordinary linear regression, and compare the results with uncalibrated model and quantile calibrator.

The code has been tested under Python 3.6.10 and Tensorflow 2.0.0 (cudatoolkit 10.0.130 + cudnn 7.6.5).

============================================================================================================

As of 2023, the backend and dependencies of this code are outdated and require some specific setup. The quickest approach is to use Anaconda:

```
conda create -n distcal python=3.6 pip
conda activate distcal
conda install tensorflow-gpu==2.0.0 tensorflow-estimator=2.0.0 numpy=1.17.0 joblib=1.0.1 scipy=1.3.0 scikit-learn matplotlib jupyter
```
