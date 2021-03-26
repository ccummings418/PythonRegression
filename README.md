# Python Regression Using Anaconda / JupyterLabs

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

-pip install xgboost


## View a Sample Code
Click on the XGBoost_ExampleCode.ipynb file above to view script to import CSV data, tune, and return model results from the XGBoost package using a fictional data set composed of various plant and reflectance parameters. 

### Tips to Help Ensure Model Works Well

#### Tip 1 - Make Sure To Remove NaN or Blank Values (or Use Interpolation to Estimate Value)
```
RawData = pd.read_csv('SampleData_XGBoost.csv')
Data = RawData.dropna()
Data

```
#### Tip 2 - Utilize One-Hot Encoding for Any Categorical Variables 
```

VarietiesOneHot = pd.get_dummies(Data.SEED_HYBRID)
Data = pd.concat([Data, VarietiesOneHot], axis=1)
Data

```

```
|SAMPLE#| 	PLANT_DENSITY| 	FERTILIZER| 	SEED_HYBRID| 	NDVI| 	BIOMASS| 	PNU| 	YIELD| 	HYB1| 	HYB2|
|0| 	1| 	35| 	100| 	HYB1| 	0.868| 	6.09| 	74.1762| 	181.879349| 	1| 	0|
|1|	2| 	35| 	200| 	HYB2| 	0.910| 	6.28| 	78.8768| 	229.975907| 	0| 	1|
|2| 	3| 	34| 	200|	HYB1| 	0.924| 	7.92| 	125.4528| 	218.899751| 	1| 	0|
|3|	4| 	34| 	100| 	HYB1| 	0.840| 	5.64| 	63.6192|	168.532993| 	1| 	0|
|4|  5| 	32| 	25| 	HYB2| 	0.714| 	2.55| 	13.0050| 	88.622588| 	0| 	1|
```

## Try XGBoost For Yourself
Download the repository and import the module into JupyterLabs to follow the XGBoost_ExampleCode.ipynb file using the synthetic data in the SampleData_XGBoost.csv file. 
