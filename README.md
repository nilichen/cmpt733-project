# Detecting Misstated Financial Statements With Deep Learning and Interactive Visualization

Katrina Ni, Leiling Tao

## Data Science Pipeline
![pipeline](https://preview.ibb.co/dQUbp7/Screen_Shot_2018_04_16_at_12_54_36_PM.png)
## How to run
### Set up environment
- Clone the repository
	```bash
	git clone https://github.com/nilichen/cmpt733-project.git
	cd cmpt733-project
	```
- Initialize the folder with virtualenv
	```bash
	virtualenv venv
	source venv/bin/activate
	```
- Install the packages
	```bash
	pip install -r requirements.txt
	```

### Download and Preprocess the data
- Download data from https://drive.google.com/open?id=1Tt2y8qn8V5oTshr9nDNtbY_hmnz1XKet
- Unzip the data and start preprocessing => will produce *annual_compustat_ratios.zip* if it does not exist in the data folder yet
    > By default it is included in the downloaded data.zip

	```
	mkdir data
	unzip data.zip -d data/
	python preprocess.py
	```

### Train the model
- See *models_with_raw_data.py*, *models_with_ratios.py*, *models_ensemble.py* for reference
 > prediction result from the mete-model is already included in downloaded data.zip as *results.csv*

### Deploy Dash application offline
- Merge preprocessed data with the results from the meta-model
	```python
	results = pd.read_csv('data/results.csv')
	df_ratios_only = pd.read_csv("data/annual_compustat_ratios.zip")
	df_ratios_only = df_ratios_only.merge(
	results[['fyear', 'gvkey', 'pred_prob']], on=['gvkey', 'fyear'])
	df_ratios_only.to_csv('data/annual_compustat_ratios.zip', index=False)
	```
-  Run in local server - see https://dash.plot.ly/deployment if want to deploy the app in Heroku
	```bash
	python app.py
	```

## The Data Product
![Dashboard](http://g.recordit.co/yeegWMh18q.gif)
