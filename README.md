# Multiple Melanoma
Daan Knoors - 11/11/22

## Objective
Develop a machine learning model for predicting the risk of fast dying or relapsing of newly diagnosed MM patients.

## Data
Download the following three data files and place under data/raw/. Keep the same file names. You might need to register an 
account first.
- **expression**: gene expression data: https://www.synapse.org/#!Synapse:syn10573789
- **clinical**: clinical data + follow-up for newly diagnosed MM patients: https://www.synapse.org/#!Synapse:syn9926878
- **dictionary**: explanation of the clinical and label annotations: https://www.synapse.org/#!Synapse:syn9744732

Other datasets and info can be found on the Multiple Myeloma Dream Challenge Site: https://www.synapse.org/#!Synapse:syn6187098/wiki/401884


## Installation
Create environment using preferred environment manager. For example using conda:
> conda env create -f environment.yml

> conda activate mm_highrisk

After activating the environment you can open jupyter notebooks using jupyter lab by running the following in the terminal:
> jupyter lab

## Project structure
```bash
mm_high_risk
|-- data/ 
|	|-- preprocessed/
|	|-- raw/	
|	|-- thesuari/															                            										                            
|
|-- docs/												
|	|-- references/										
|
|-- models/											    
|
|-- notebooks/											
|	|-- 000_exploratory_data_analysis.ipynb				
|	|-- 001_train_model.ipynb							
|	|-- 002_evaluate_model.ipynb							
|
|-- src/											    
|	|-- __init__.py                                     
|	|-- config.py                                       
|	|-- gene.py                                       	
|	|-- model.py                                        
|	|-- preprocess.py			
|	|-- utils.py                                      
|	|-- visual.py	                                    
|
|-- environment.yml										
|-- README.md
```