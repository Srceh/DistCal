# DistCal
Code for the ICML 2019 paper: Distribution Calibration for Regression

This implementation uses Tensorflow 2.0 (https://www.tensorflow.org/) as backend for automatic differentiation and GPU acceleration.

example.ipynb contains the example code of applying the GP-Beta approach on the Boston dataset and ordinary linear regression, and compare the results with uncalibrated model and quantile calibrator.

The code has been tested under Python 3.6.10 and Tensorflow 2.0.0 (cudatoolkit 10.0.130 + cudnn 7.6.5).
