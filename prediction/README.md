# prediction

## What's here:
### script
- `prediction.ipynb`: notebook providing visual outline of the prediction process
- `prod.py`: script to predict activity of the test set using the training set

### output
- `test_pred.csv`: compounds from the test set and associated model predictions
- `model.pkl`: file storing the model object
- `AUC_curve.csv`: area under the curve (AUC) of the associated precision-recall curve for the model

## Procedure
We provide some examples to demonstrate how this method can be applied to predict whether untested compounds will bind. A more visual example of the process is shown in the notebook `prediction.ipynb`. A script version of the prediction method can also be run with the following command: 
```python
# Run prediction with randomized train/test split
python prediction_script.py --seed ${SEED} --frac ${TRAIN_FRAC}
```
We provide options for saving the results of the prediction method as well as the model used. 
