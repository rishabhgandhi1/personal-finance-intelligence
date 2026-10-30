Personal Finance Intelligence

An end-to-end personal finance analytics application that transforms transaction data into actionable spending insights using data analytics, machine learning, statistical forecasting, and generative AI.

Overview

Managing personal finances involves more than tracking income and expenses. Understanding spending patterns, identifying unusual expenses, predicting future spending, and interpreting financial summaries can help individuals make better-informed financial decisions.

Personal Finance Intelligence is a Python-based application designed to address these everyday problems. It processes transaction data, stores financial records in a relational database, analyzes spending behaviour, predicts expense categories, forecasts monthly expenses, and generates plain-English financial explanations using an LLM.

Dataset note: The included transaction dataset is synthetically generated for development and demonstration. Results from this dataset should not be interpreted as evidence of performance on real-world banking data.

Key Features

1. Transaction Ingestion and Processing

* Import transaction records from CSV files.
* Validate transaction fields, dates, amounts, and categories.
* Handle duplicate transaction IDs and invalid records.
* Clean and transform transaction data using Pandas and NumPy.
* Engineer time-based features for downstream analysis.

2. Financial Analytics

* Calculate total income, expenses, and net balance.
* Analyze spending by category.
* Generate monthly income and expense summaries.
* Identify spending patterns using statistical analysis.

3. Machine Learning Expense Categorization

* Predict expense categories from transaction descriptions.
* Use TF-IDF for text feature extraction.
* Train a Logistic Regression classification model.
* Evaluate performance using stratified random splitting and unseen-description splitting.

Evaluation results on the included synthetic dataset:

Evaluation method	Accuracy
Stratified random split	100.00%
Unseen-description split	39.93%

The random split score is inflated by repeated transaction descriptions. The unseen-description evaluation is a more demanding test of generalization. The results demonstrate why evaluation design matters and why performance on synthetic data should not be presented as real-world model performance.

4. Statistical Analysis and Expense Forecasting

* Analyze monthly spending variability.
* Compare spending across two six-month periods using Welch’s t-test.
* Forecast future monthly expenses using Statsmodels Exponential Smoothing.
* Evaluate forecasts using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).
* Compare the forecasting model against a last-observation baseline.

The baseline performed better than the Exponential Smoothing model on the current three-month holdout. This limitation is retained rather than presenting the more complex model as automatically superior.

5. Experiment Tracking

* Track machine learning experiments using MLflow.
* Record model parameters, accuracy, training and testing row counts.
* Store classification reports.
* Compare different evaluation strategies.

6. Interactive Dashboard and AI Financial Insights

* Explore financial summaries through a Streamlit dashboard.
* Visualize category-wise spending and monthly trends.
* Predict expense categories.
* View monthly expense forecasts.
* Generate plain-English explanations of verified financial summaries using Google’s Gemini API.

The LLM explains calculated figures; it does not independently calculate or verify financial records.

Technology Stack

Area	Technologies
Programming	Python
Data Processing	Pandas, NumPy
Database	SQLite, SQLAlchemy
Machine Learning	Scikit-learn
Statistical Analysis	SciPy, Statsmodels
Visualization	Matplotlib, Seaborn
Experiment Tracking	MLflow
Backend API	FastAPI, Uvicorn
Frontend	Streamlit
Generative AI	Google Gemini API
Testing	Pytest, HTTPX
Containerization	Docker, Docker Compose
Version Control	Git, GitHub

Architecture

flowchart TD
    A[CSV Transaction Data] --> B[Data Ingestion and Validation]
    B --> C[Data Processing and Feature Engineering]
    C --> D[(SQLite Database)]
    D --> E[Financial Analytics]
    D --> F[ML Expense Categorization]
    E --> G[Statistical Analysis and Forecasting]
    F --> H[MLflow Experiment Tracking]
    E --> I[FastAPI Backend]
    F --> I
    G --> I
    I --> J[Streamlit Dashboard]
    I --> K[Gemini AI Financial Explanations]

Project Structure

personal-finance-intelligence/
├── data/
│   ├── processed/
│   └── sample/
├── docs/
├── models/
├── notebooks/
├── scripts/
├── src/
│   ├── analytics/
│   ├── api/
│   ├── database/
│   ├── ingestion/
│   ├── ml/
│   ├── processing/
│   └── services/
├── streamlit_app/
├── tests/
├── .env.example
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── requirements.txt

API Endpoints

The FastAPI backend exposes the following endpoints:

Method	Endpoint	Description
GET	/	API information
GET	/health	API and database health
GET	/summary	Overall financial summary
GET	/spending/categories	Category-wise spending
GET	/spending/monthly	Monthly income and expenses
POST	/predict/category	Predict an expense category
GET	/forecast/monthly	Forecast monthly expenses
GET	/explain/finances	Generate an AI explanation of financial summaries

Interactive API documentation is available at /docs when the API is running.

Installation and Configuration

Prerequisites

* Python 3.13
* Git
* Docker Desktop (for containerized execution)
* A Google Gemini API key for AI financial explanations

1. Clone the repository

git clone <YOUR_GITHUB_REPOSITORY_URL>
cd personal-finance-intelligence

2. Create and activate a virtual environment

python3 -m venv .venv
source .venv/bin/activate

3. Install dependencies

python -m pip install -r requirements.txt

4. Configure environment variables

cp .env.example .env

Add your Gemini API key to .env:

GEMINI_API_KEY=your_api_key_here

Keep .env private. It must never be committed to GitHub.

5. Initialize the database and import sample data

Initialize the database and import the included sample transactions:

python -m scripts.import_sample_data

This command creates the required database tables and imports data/sample/transactions.csv.

The ingestion process validates records and prevents duplicate transaction IDs from being inserted again.

Running with Docker

Build and start the application:

docker compose build
docker compose up -d

Access the application:

* Streamlit Dashboard: http://localhost:8501
* FastAPI: http://localhost:8000
* API Documentation: http://localhost:8000/docs

Check running containers:

docker compose ps

View application logs:

docker compose logs -f

Stop the application:

docker compose down

Running Tests

Run the complete test suite:

python -m pytest -v

The project has been tested with 69 passing tests across database operations, transaction ingestion, data processing, analytics, machine learning, and API functionality.

Visualizations

The project includes generated visualizations for:

* Monthly income versus expenses
* Category-wise spending
* Monthly spending trends

These charts are available in data/sample/.

Limitations and Future Considerations

* The included dataset is synthetic and does not represent actual bank transactions.
* Expense categorization performance on unseen descriptions is limited; broader real-world data is needed for meaningful evaluation.
* Forecast evaluation currently uses a small chronological holdout. More historical observations are needed for stronger conclusions.
* Financial explanations are informational and should not be treated as professional financial advice.
* The application is a portfolio project, not a production banking or financial-management service.

Project Objective

This project demonstrates an end-to-end applied data science workflow: from raw transaction ingestion and data validation to analytics, machine learning, statistical evaluation, experiment tracking, API development, interactive visualization, LLM integration, automated testing, and containerization.