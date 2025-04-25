#   Heart Disease Risk Analysis

##   Project Overview

This project focuses on analyzing patient health data to understand and explore factors related to heart disease risk. The primary goal was to gain insights from the data through exploratory data analysis (EDA), data preprocessing, and feature engineering, with machine learning models used as a tool for analysis.

##   Dataset

The dataset used in this project is the Kaggle heart disease dataset, containing 10,000 patient records with 21 features.

* Source: Kaggle
* Size: 10,000 rows x 21 features
* Target Variable: Heart.Disease.Status (Binary classification)
* Features: Includes numerical (e.g., Age, Blood Pressure) and categorical (e.g., Smoking, Alcohol Consumption) variables.

##   Data Analysis and Processing

The following data analysis and processing techniques were applied:

* Data cleaning: Handling missing values and inconsistencies.
* Feature engineering: Transforming categorical features (label encoding) and scaling numerical features (standardization).
* Exploratory Data Analysis (EDA): Visualizing data distributions and relationships using histograms, box plots, bar plots, and pie charts.
* Feature selection: Identifying relevant features using statistical (Logistic Regression) and model-based (Random Forest) techniques.

##   Machine Learning Models

The following machine learning models were used for classification and analysis:

* K-Nearest Neighbors (KNN)
* Logistic Regression
* Random Forest
* XGBoost

The primary focus was on using these models to understand feature importance and relationships within the data, rather than optimizing them for predictive accuracy.

##   Files

* `code/heart_disease_analysis.R`: R script for data analysis, processing, and modeling.
* `data/heart_disease.csv`: The heart disease dataset.
* `report/heart_disease_report.pdf`: Project report detailing the data analysis process and findings.

##   Dependencies

* R (version 4.4.2)
* The following R packages:
    * `tidyverse`
    * `caret`
    * `ggplot2`
    * `dplyr`

##   Usage

1.  Clone the repository.
2.  Install the required R packages.
3.  Run the R script:  `Rscript code/heart_disease_analysis`

##   Key Insights

Refer to the project report (`report/heart_disease_report`) for detailed insights and visualizations.

##   Limitations

* The project's emphasis was on data analysis, and model optimization was not the primary focus.
* The findings are based on a single dataset and may not be generalizable to other populations.

##   Authors

* Batoul Ballout - batoulballout96@gmail.com
* Rasha Harb - rashah.harb@gmail.com
* Razan Doughman - razan.doughman@gmail.com
