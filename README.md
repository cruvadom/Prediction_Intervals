# Computing Calibrated Prediction Intervals
Computing calibrated prediction intervals for neural network regressors, according to the paper:
"Calibrated Prediction Intervals for Neural Network Regressors"
by Gil Keren, Nicholas Cummins and Björn Schuller. 

The standard output of a regressor model is a point estimate for the label. [Prediction intervals](https://en.wikipedia.org/wiki/Prediction_interval) are intended to be contain information about the uncertainty in the regressor's prediction. Specifically, these are estimates of the interval in which the target label is expected to lie with a prescribed probability. For example, a point estimate may be the single scalar 4.2, as an estimation of the unknown label. A prediction interval (2.1, 6.3) with a confidence level of 80% may indicate that with 80% probability, the label will be between 2.1 and 6.3. 

A set of prediction intervals is considered calibrated if the prescribed confidence level (80% in this example) reflects the actual probability that labels lie in the prediction intervals. For example, assume we produce a 1000 prediction intervals with 80% confidence level, for 1000 different examples. If only 60% of the prediction intervals contain the actual label of their repective example, this set of prediction intervals is considerned not properly calibrated. 

To compute prediction intervals for neural network regressors, the real-valued labels are binned into N classes, and the network is trained as a neural network classifier. The outputs of the classifier are the class logits - the activations of the top layer just before the softmax normalization. The main usage of the code in this repository is to compute calibrated prediction intervals, based on the output logits from a neural network. 

Main functions:
`get_calibrated_pis(algorithm, confidence_level, validation_logits, validation_real_labels, test_logits, bins, centers)`
is the main function of the code. It implements the Temperature Scaling and Empirical Calibration algorithms from the paper "Calibrated Prediction Intervals for Neural Network Regressors". It chooses the algorithm hyperparameters using the validation set logits and real-valued labels, and uses these hyperparamters to compute prediction intervals for the test set with a given confidence level.
