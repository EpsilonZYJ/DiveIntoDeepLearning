from autogluon.tabular import TabularDataset, TabularPredictor

train_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/titanic/train.csv')
id, label = 'PassengerId', 'Survived'
predictor = TabularPredictor(label=label).fit(train_data.drop(columns=[id]))

import pandas as pd

test_data = TabularDataset('test.csv')
preds = predictor.predict(test_data.drop(columns=[id]))
submission = pd.DataFrame({id: test_data[id], label: preds})
submission.to_csv('submission.csv', index=False)