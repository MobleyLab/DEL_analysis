# prediction

## What's here:
### script
- `prediction_holdout_compounds.ipynb`: notebook with proof of concept for using P(active) to predict trisynthon binding
- `prediction.ipynb`: notebook providing visual outline of the prediction process involving unknown building blocks
- `prod.py`: script to predict activity of the test set trisynthons containing unknown building blocks

## Procedure
We provide some examples to demonstrate how this method can be applied to predict whether untested compounds will bind. The `prediction_holdout_compounds.ipynb` notebook applies a decision tree model to test data consisting of new combinations of known building blocks. All building blocks in the test set are found in the training set, so P(active) values for each building block are roughly known. 

For predictions involving untested building blocks, we provide two separate code files. A more visual example of the process is shown in the notebook `prediction.ipynb`. A script version of the prediction method can also be run with the following command: 
```python
# Run prediction with randomized train/test split
python prediction_script.py --seed ${SEED} --frac ${TRAIN_FRAC}
```
We provide options for saving the results of the prediction method and the trained model. 
