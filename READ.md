MatchMaster - Predictive Cricket Pipeline

A predictive cricket match pipeline built using modern ML techniques

Overview
MatchMaster is an end-to-end machine learning pipeline for predicting cricket match outcomes. It integrates multiple historical datasets including batsman-level, bowler-level, and match-level scorecards along with your own dataset. The pipeline computes over 25 features based on the historical performance data, handles missing values, splits the dataset for training and testing, trains a dynamic blended model using ensemble methods, and finally evaluates the model using accuracy and F1 score.

The blended model aggregates predictions from several classifiers – GBM, LightGBM, XGBoost, and CatBoost – with dynamic weight assignment based on individual model performance.

Features
Data Ingestion: Reads four essential datasets:

Batsman Level Scorecard: Contains detailed performance of batsmen.

Bowler Level Scorecard: Contains detailed performance of bowlers.

Match Level Scorecard: Provides historical match data and contextual information.

Your Dataset: The main dataset containing match-level features for training.

Feature Engineering:
Calculates approximately 25 features using historical data. These include:

Toss-related binary features (e.g., toss_winner_01, toss_decision_01).

Batting performance metrics such as average runs, strike rate, and head-to-head opponent metrics.

Bowling performance metrics including economy rates (top 4 & bottom 4), bowler form scores, wicket counts, and more.

Comparative (ratio) features that capture differences between Team 1 and Team 2.

Other domain-specific features derived from historical trends (e.g., last 5 match averages, venue-specific performance, etc.).

Missing Value Handling:
Provides strategies for imputing missing values in both numerical and categorical features.

Data Splitting:
Splits data into training and testing sets using configurable parameters.

Blended Model Training:
Trains an ensemble (blended) model that combines predictions from:

Gradient Boosting Machine (GBM)

LightGBM

XGBoost

CatBoost
The model assigns dynamic weights to each classifier based on validation accuracy, helping to boost overall performance.

Model Evaluation:
Evaluates the trained model on the test set using accuracy and F1 score metrics, allowing you to assess predictive performance.

Repository Structure
bash
Copy
Edit
MatchMaster-Predictive-Cricket-Pipeline/
├── data/
│   ├── batsman_level_scorecard.csv
│   ├── bowler_level_scorecard.csv
│   ├── match_level_scorecard.csv
│   └── your_dataset.csv
├── src/
│   ├── feature_engineering.py          # Contains feature engineering strategy classes
│   ├── model_building.py               # Contains blended model building logic
│   └── ...                             # Other modules (e.g., data splitting, hyperparameter optimization)
├── steps/
│   ├── data_ingestion_step.py          # Code to ingest CSV data
│   ├── data_splitter_step.py           # Splits data for training and testing
│   ├── feature_engineering_step.py     # Orchestrates feature engineering
│   ├── handle_missing_values_step.py   # Missing value handling
│   ├── hyperparameter_optimization_step.py  # Hyperparameter tuning using Optuna
│   └── model_evaluator_step.py         # Evaluates model performance
├── training_pipeline.py                # Main pipeline that ties all steps together
├── README.md                           # This file
└── requirements.txt                    # Required packages and versions
Installation
Clone the Repository:

bash
Copy
Edit
git clone https://github.com/vijay2903/MatchMaster-Predictive-Cricket-Pipeline.git
cd MatchMaster-Predictive-Cricket-Pipeline
Create and Activate a Virtual Environment:

bash
Copy
Edit
python -m venv venv
source venv/bin/activate   # On Windows use: venv\Scripts\activate
Install the Requirements:

bash
Copy
Edit
pip install -r requirements.txt
Usage
Run the main pipeline by executing the training_pipeline.py script:

bash
Copy
Edit
python training_pipeline.py
This script will:

Ingest the necessary CSV files.

Perform feature engineering by computing domain-specific features.

Handle missing values.

Split the data into training and test sets.

Optimize hyperparameters for a blended model that uses GBM, LightGBM, XGBoost, and CatBoost.

Evaluate the model's performance on the test set and log the evaluation metrics.

Configuration
Paths:
Ensure the paths to your CSV files (batsman, bowler, match, and training data) in training_pipeline.py are correct for your environment.

Hyperparameters:
You can configure the number of trials and CV folds for hyperparameter optimization in hyperparameter_optimization_step.py.

Feature Selection:
Modify the list of columns in the columns_to_keep parameter in training_pipeline.py to customize the final set of features used for training.

Contribution
Feel free to open issues or submit pull requests if you have suggestions, bug fixes, or improvements.

License
This project is licensed under the MIT License.

Acknowledgments
Built with ZenML, a MLOps framework that integrates well with MLflow and other tools.

Inspired by domain-specific cricket analytics and advanced ensemble machine learning techniques.

