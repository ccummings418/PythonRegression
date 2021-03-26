# Python Regression Using XGBoost

*These are some useful coding modules I have used for XGBoost regression modeling.*

## Regression Modeling
### Reason for XGBoost?
There are many different machine learning algorithms that are used for supervised regression modeling that range in complexity from Linear Regression to more advanced models such as Random Forest, SVM. eXtreme Gradient Boosting (XGBoost) is an additional learning model which has demonstrated excellent performance for use in supervised modeling and has won many data science / machine learning competitions. The main advantage of XGBoost is its vast number of hyperparameters that can be tuned for either large or small data sets. Several of which are learning rate (eta), maximum depth of tree (max_depth), minimum weight until further partitioning (min_child_weight), conservativeness of model (gamma), and L1 (alpha) as well as L2 (lambda) regulation. 

## Reference of original XGBoost manuscript
Tianqi Chen and Carlos Guestrin. XGBoost: A Scalable Tree Boosting System. In 22nd SIGKDD Conference on Knowledge Discovery and Data Mining, 2016
PDF https://arxiv.org/pdf/1603.02754.pdf

Additional documentation:

https://xgboost.readthedocs.io/en/latest/index.html

## pypi website for installing XGBoost in Anaconda / Python
https://pypi.org/project/xgboost/

### Install tips
Using conda, either use 

- conda install -c conda-forge xgboost

or

- pip install xgboost


## View a Sample Code
Click on the XGBoost_ExampleCode.ipynb file above to view script to import CSV data, tune, and return model results from the XGBoost package using a fictional data set composed of various plant and reflectance parameters. 

### Helpful Tips to Ensure Model Works Well

#### Tip 1 - Make Sure To Remove NaN or Blank Values (or Use Interpolation to Estimate Value)
```
RawData = pd.read_csv('SampleData_XGBoost.csv')
Data = RawData.dropna()
Data
```
#### Tip 2 - Utilize One-Hot Encoding for Any Categorical Variables 
XGBoost and most other machine learning models have a difficult time using categorical variables, so there are several techniques for making the data easier to use. If the attribute shows a linear range of values (i.e. low, medium, high) it can be converted numerically into a single column wiht 1,2,and 3. However, if the column is categorical like crop type, hybrid variety, or soil type, it is likely better to use one-hot encoding which creates unique columns of 1's or 0's for each type of attribute, with 1 meaning TRUE and 0 meaning FALSE for being that attribute. 

```
VarietiesOneHot = pd.get_dummies(Data.SEED_HYBRID)
Data = pd.concat([Data, VarietiesOneHot], axis=1)
Data
```

|SAMPLE#| 	PLANT_DENSITY| 	FERTILIZER| 	SEED_HYBRID| 	HYB1| 	HYB2|
--- | --- | --- | --- |--- |--- |
|1| 	35| 	100| 	HYB1| 	 	1| 	0|
|2| 	35| 	200| 	HYB2| 	 	0| 	1|
|3| 	34| 	200|	HYB1| 	 	1| 	0|
|4| 	34| 	100| 	HYB1| 		1| 	0|
|5| 	32| 	25| 	HYB2| 	 	0| 	1|

#### Tip 3 - XGBoost Uses DMatrix to Cluster Testing and Training Data 
XGBoost has a unique format for dataframes which uses its own DMatrix format, which improves performance and makes it easier to fit the model.
```
DMatrix_train = xgb.DMatrix(X_train,y_train)
DMatrix_test = xgb.DMatrix(X_test, y_test)
```
#### Tip 4 - Utilize Hyperparameter Tuning and XGBoost Cross Validation Package (xgb.cv) To Best Configure Models
As mentioned before, XGBoost has a variety of different hyperparameters to tune which can optimize the resulting models for testing data. The specific parameters selected here are max_depth, min_child_weight, and learning rate (eta). 

XGBoost also has a helpful cross-validation (cv) parameter which can be used within the training dataset to further improve finding the correct hyperparameters to use. In addition, it utilizes a early_stopping_rounds parameter to further alleviate the likelihood of overfitting a model based on training data. 

```
results = xgb.cv(dtrain = DMatrix_train, params = params, nfold=3,num_boost_round=10000, early_stopping_rounds=3, metrics="mae",seed=123)
```

#### Tip 5 - Plot the Predicted versus Measured Data to Visualize How Model is Performing

![index](https://user-images.githubusercontent.com/80427122/112690783-bfb98100-8e52-11eb-95fd-fee72b18ab8b.png)

## Try XGBoost For Yourself
Download the repository and import the module into JupyterLabs to follow the XGBoost_ExampleCode.ipynb file using the made-up data set in the SampleData_XGBoost.csv file. 
