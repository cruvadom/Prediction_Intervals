####################################################3####################
# Code for the paper:
# "Calibrated Prediction Intervals for Neural Network Regressors"
# by Gil Keren, Nicholas Cummins, Bjoern Schuller. 
# 
# Computes calibrated prediction invervals for neural network regressors.
#########################################################################


import numpy as np

def softmax_numpy(logits, temperature=1.0, exp_base=np.e):
  minus_max = logits - logits.max(axis=1).reshape([len(logits), 1])
  if exp_base == np.e:
    step1 = np.exp(minus_max / temperature)
  else:
    step1 = np.power(exp_base, minus_max / temperature)
  step2 = step1.sum(axis=1)
  step3 = step1 / step2.reshape([len(step2), 1])
  return step3
    
    
def get_intervals(probs, confidence, bins, centers):
  nclasses = len(bins) - 1
  nex = len(probs)
  
  # Calculate predictions
  cont_preds = (probs * centers).sum(axis=1)
  class_preds = np.digitize(cont_preds, bins) - 1
  assert class_preds.min() >= 0
  
  solved = np.zeros(dtype=np.bool, shape=[nex])
  neighborhood = 0
  s, e = np.zeros(dtype=np.int32, shape=[nex]), np.zeros(dtype=np.int32, shape=[nex])
  while not np.all(solved):
    # Get the neighborhood probability
    start_cols = np.maximum(0, class_preds - neighborhood)
    end_cols = np.minimum(nclasses - 1, class_preds + neighborhood)
    mask = np.array((range(0, nclasses) * nex)).reshape([nex, nclasses])
    mask = np.logical_and(mask >= start_cols.reshape((-1, 1)), mask <= end_cols.reshape((-1, 1)))
    mask = np.logical_not(mask)
    summed_cols = np.array(np.ma.masked_array(probs, mask).sum(axis=1))
    
    # Save the newly solved
    readynow = summed_cols >= confidence
    newly_solved = np.logical_and(readynow, np.logical_not(solved))
    s[newly_solved], e[newly_solved] = start_cols[newly_solved], end_cols[newly_solved]
    solved = np.logical_or(solved, newly_solved)
    neighborhood += 1

  # Convert back to cont values
  lefts, rights = bins[:-1], bins[1:]
  s, e = lefts[s], rights[e]
  return s, e


def evaluate_intervals(starts, ends, label):
  actual_confidence = np.logical_and(label >= starts, label <= ends).mean()
  return actual_confidence


def find_empirical(logits, real_labels, confidence, bins, centers, accuracy=0.001):
  probs = softmax_numpy(logits)
  
  # Set the search lower and upper limits
  minval, maxval = 0.0001, 1
  
  # Start the binary search
  t = -1
  found = False
  while not found:
    # Set new t
    last_t = t
    t = (minval + maxval) / 2.0
    if np.abs(last_t - t) < 0.001:
      break
     
    s, e = get_intervals(probs, t, bins, centers)
    actual_confidence = evaluate_intervals(s, e, real_labels)
    
    error = np.abs(confidence - actual_confidence)
    if error < accuracy:
      found = True
    if actual_confidence > confidence:
      maxval = t
    else:
      minval = t
      
  return t
    
    
def find_best_temperature(logits, real_labels, confidence, bins, centers, accuracy=0.001): 
  
  # Set the search lower and upper limits
  minval, maxval = 0.0001, 10
  
  # Start the binary search
  t = -1
  found = False
  while not found:
    # Set new t
    last_t = t
    t = (minval + maxval) / 2.0
    if np.abs(last_t - t) < 0.001:
      break
    
    # Compute intervals
    probs = softmax_numpy(logits, temperature=t)
    s, e = get_intervals(probs, confidence, bins, centers)
    actual_confidence = evaluate_intervals(s, e, real_labels)
    
    # Change search limits
    error = np.abs(confidence - actual_confidence)
    if error < accuracy:
      found = True
    if actual_confidence > confidence:
      maxval = t
    else:
      minval = t
      
  return t


r"""
Given real-values labels and the bin edges, converts to class labels.  
"""
def create_class_labels(real_labels, bins):
  return np.digitize(real_labels, bins) - 1


r"""
Given the range of the real-valued labels and the number of desired classes, 
creates the bins and the centers of the classes. The bins are created to be
equally spaced in the minval-maxval range, where bins[0] = -np.inf and 
bins[-1] = np.inf. The centers are created to be in the center of the classes
range (before setting first and last boundaries to +/- infinity). 
A good choice of values is such that minval and maxval reflect the minimum 
and maximum real-values label values in the training set.    
  Args:
    minval: minimal value of real-valued label to consider. 
    maxval: maximal value of real-valued label to consider.
    nclasses: number of desired classes.
      
  Returns:
    bins, centers: bins is the boundary of the classes, centers is the
    expected prediction for each class.  
  """
def prepare_bins_and_centers(minval, maxval, nclasses):
  bins = np.linspace(minval, maxval, nclasses + 1)
  centers = (bins[1:] + bins[:-1]) / 2.0
  bins[0], bins[-1] = -np.inf, np.inf
  return bins, centers


r"""
Computes prediction intervals for the test set, according to the paper:
"Calibrated Prediction Intervals for Neural Network Regressors"
by Gil Keren, Nicholas Cummins, Bjoern Schuller.
 
First, the algorithm hyperparameters are chosen on the validation set, then
the chosen parameters are used to compute prediction intervals on the 
test set.   
  Args:
    algorithm: Calibration algorithm. Either 'temperature' for Temperature Scaling
      or 'empirical' for Empirical Calibration.
    confidence_level: The confidence level for the prediction intervals. 
      In the range (0,1). 
    validation_logits: logits for the validation set. A numpy array 
      of shape [n_examples, n_classes].
    validation_real_labels: real-values labels for the validation set. 
      A numpy array of shape [n_examples].
    test_logits: logits for the test set. A numpy array 
      of shape [n_examples, n_classes].
    bins: The boundaries of the classes. For example bins of [-np.inf, 1, 2, np.inf]
      represents 3 classes: (-np.inf, 1], (1, 2], (2, -np.inf). A numpy array of
      shape [n_classes + 1]
    centers: The centers of the classes. For each class, one value that represents 
      the expected prediction for this class.
      
  Returns:
    s, e: two numpy arrays each of shape [n_examples] with the starts
    and the ends of the calibrated prediction intervals for the desired
    confidence level. 
  """
def get_calibrated_pis(algorithm, confidence_level, validation_logits, validation_real_labels, test_logits, bins, centers):
  assert bins[0] == -np.inf and bins[-1] == np.inf
  
  if algorithm == 'temperature':
    # Find best temperature based on the validation set
    print('Finding best temperature based on the validation set...')
    best_t = find_best_temperature(validation_logits, validation_real_labels, confidence_level, bins, centers)
    print('Best temperature on the validation set: {}'.format(best_t))
    
    # Use this temperature to get calibrated prediction intervals for the test set
    print('Getting calibrated prediction intervals for the test set...')
    s, e = get_intervals(softmax_numpy(test_logits, temperature=best_t), confidence_level, bins, centers)
    print('Done.')
    print('')
    return s, e
  
  elif algorithm == 'empirical':
    # Finding empirical confidence level based on the validation set
    print('Finding empirical confidence level based on the validation set...')
    calibrated_conf_level = find_empirical(validation_logits, validation_real_labels, confidence_level, bins, centers)
    print('Empirical calibrated confidence level base on validation set: {}'.format(calibrated_conf_level))
    
    # Use the empirical confidence levelto get calibrated prediction intervals for the test set
    print('Getting calibrated prediction intervals for the test set...')
    s, e = get_intervals(softmax_numpy(test_logits), calibrated_conf_level, bins, centers)
    print('Done.')
    print('')
    return s, e
  
  else:
    print('Algorithm not supported. Use mode = temperature/empirical')


if __name__ == '__main__':
  
  # Part 0: Preparations
  # Load the validation set logits, validation set logits, test set labels, 
  # bins (class boundaries), centers (class centers).
  # If there is access to the test set labels (for part 3), can load them as well.
  # All the above mentioned should be numpy matrices. 

  # Shapes: 
  # validation / test logits: [n_example, n_classes]
  # validation / test real labels: [n_exmaples]
  # bins: [n_classes + 1]
  # centers: [n_classes] 
  
  # These are some sample logits and labels. When using this code, 
  # replace with the actual logits and labels.
  N_EXAMPLES = 10000
  N_CLASSES = 20
  bins, centers = prepare_bins_and_centers(0.0, 1.0, N_CLASSES)
  validation_logits = np.random.uniform(low=0.0, high=10.0, size=[N_EXAMPLES, N_CLASSES])
  validation_real_labels = np.random.uniform(low=0.0, high=1.0, size=[N_EXAMPLES])
  test_logits = np.random.uniform(low=0.0, high=10.0, size=[N_EXAMPLES, N_CLASSES])
  test_real_labels = np.random.uniform(low=0.0, high=1.0, size=[N_EXAMPLES])
  
  # Part 1: Demonstration of calibration on the validation set
  # This part demonstrates Temperature Scaling and 
  # Empirical Calibration on the validation set.   
  # It is not needed for real life applications.
  confidence_level = 0.8
  print('Demonstrating calibration on the validation set:')
  starts, ends = get_intervals(softmax_numpy(validation_logits), confidence_level, bins, centers)   
  actual_confidence = evaluate_intervals(starts, ends, validation_real_labels)
  print('With no calibration: desired confidence level {}, acutal confidence level {}'.format(confidence_level, actual_confidence))
  best_t = find_best_temperature(validation_logits, validation_real_labels, confidence_level, bins, centers)
  starts, ends = get_intervals(softmax_numpy(validation_logits, temperature=best_t), confidence_level, bins, centers)
  actual_confidence = evaluate_intervals(starts, ends, validation_real_labels)
  print('With Temperature Scaling: desired confidence level {}, acutal confidence level {}'.format(confidence_level, actual_confidence))
  calibrated_conf_level = find_empirical(validation_logits, validation_real_labels, confidence_level, bins, centers)
  starts, ends = get_intervals(softmax_numpy(validation_logits), calibrated_conf_level, bins, centers)
  actual_confidence = evaluate_intervals(starts, ends, validation_real_labels)
  print('With Empirical Calibration: desired confidence level {}, acutal confidence level {}'.format(confidence_level, actual_confidence))
  print('')
  
  # Part 2; Get prediction intervals on the test set.
  # This part does the whole procedure - estimates the algorithm 
  # parameters based on validation set and use these parameters 
  # to compute prediction intervals on the test set. This is the 
  # realistic scenario.
  # Change algorithm to switch between Temperature Scaling and 
  # Empirical Calibration.
  confidence_level = 0.8
  algorithm = 'temperature' # 'temperature' / 'empirical'
  starts, ends = get_calibrated_pis(algorithm, confidence_level, validation_logits, 
                                    validation_real_labels, test_logits, bins, centers)
  
  # Part 3: Evalute the calibration error on the test set. 
  # This can be done only if there is access to the test set labels. 
  actual_confidence = evaluate_intervals(starts, ends, test_real_labels)
  print('Test evaluation: desired confidence level {}, acutal confidence level {}'.format(confidence_level, actual_confidence))
  
