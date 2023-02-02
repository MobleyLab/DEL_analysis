# prediction

## What's here:
### input
    - `train_example.csv`: file of compounds selected as a training set
    - `test_example.csv`: file of compounds selected as a test set

### script
    - `prod.py`: script to predict activity of the test set using the training set
    - `train_test.py`: script to split a dataset into train/test sets

### output
    - `test_pred.csv`: compounds from the test set and associated model predictions
    - `model.pkl`: file storing the model object
    - `AUC_curve.csv`: area under the curve (AUC) of the associated precision-recall curve for the model
