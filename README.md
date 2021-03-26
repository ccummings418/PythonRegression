# Python Regression Using Anaconda / JupyterLabs

*These are some useful coding modules I have used for XGBoost regression modeling.*

## Regression Modeling
### Reason for XGBoost?
There are many different machine learning algorithms that are used for supervised regression modeling that range in complexity from Linear Regression to more advanced models such as Random Forest, SVM. eXtreme Gradient Boosting (XGBoost) is an additional learning model which has demonstrated excellent performance for use in supervised modeling and has won many data science / machine learning competitions. The main advantage of XGBoost is its vast number of hyperparameters that can be tuned for either large or small data sets. 

## pypi website for installing XGBoost in Anaconda / Python
https://pypi.org/project/xgboost/

### Install tips
Using conda, either use 

- conda install -c conda-forge xgboost

or

-pip install xgboost

## Reference of original XGBoost manuscript
Tianqi Chen and Carlos Guestrin. XGBoost: A Scalable Tree Boosting System. In 22nd SIGKDD Conference on Knowledge Discovery and Data Mining, 2016
PDF https://arxiv.org/pdf/1603.02754.pdf

Additional documentation:

https://xgboost.readthedocs.io/en/latest/index.html

## View a Sample Code
Click on the XGBoost_ExampleCode.ipynb file above to view script to import CSV data, tune, and return model results from the XGBoost package using a fictional data set composed of various plant and reflectance parameters. 

## Try XGBoost For Yourself
Download the repository and import the module into JupyterLabs to follow the XGBoost_ExampleCode.ipynb file using the synthetic data in the SampleData_XGBoost.csv file. 
