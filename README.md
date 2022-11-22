# Federated Learning - Titanic

## Objective
Train a ML model to predict the survival probability in a federated setup and benchmark it against a model being trained on the full dataset.

## Data
- **Titanic**: split in two sets A and B with same feature input but different records, i.e. horizontally partitioned data.
- **Test**: test data with same features as titanic but without target column 'pred'

Target class: 'Survived'

Source of data: https://www.kaggle.com/competitions/titanic/data

## Installation
Create environment using preferred environment manager. For example using conda:
> conda env create -f environment.yml

> conda activate fl_titanic

After activating the environment you can open jupyter notebooks using jupyter lab by running the following in the terminal:
> jupyter lab

## Project structure
```bash
fl_titanic
|-- data/ 
|	|-- preprocessed/
|	|-- raw/	
|								
|-- notebooks/											
|	|-- 000_exploratory_data_analysis.ipynb				
|	|-- 001_train_model.ipynb							
|
|-- src/											    
|	|-- __init__.py
|	|-- aggregation.py    	  : federated output aggregration strategies       
|	|-- client.py             : client info and methods to perform computations on client data    
|	|-- collaboration.py      : federation of clients and authorized algorithms                                   
|	|-- config.py             : project specfic variables and class instances
|	|-- data.py           	  : client data pointers
|	|-- model.py          	  : classifiers       
|	|-- preprocess.py         : data preprocessing methods
|	|-- server.py         	  : orchestration of algorithms between clients 
|	|-- stats.py          	  : statistical functions to compute on local data                      
|	|-- utils.py              : utility functions                                    
|	|-- visual.py             : visualization methods                                    
|
|-- .gitignore                                  
|-- environment.yml										
|-- README.md
```