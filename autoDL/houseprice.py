from autogluon.tabular import TabularDataset, TabularPredictor
import numpy as np

train_data = TabularDataset('../data/train.csv')
id, label = 'Id', 'Sold Price'

large_val_cols = ['Lot', 'Total interior livable area', 'Tax assessed value', 'Annual tax amount', 'Listed Price', 'Last Sold Price']

for c in large_val_cols + [label]:
    train_data[c] = np.log(train_data[c]+1)

predictor = TabularPredictor(label=label).fit(train_data.drop(columns=[id]))