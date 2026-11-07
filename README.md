# FinWise AI

FinWise AI is a Flask personal finance platform intended to turn a user's own transaction history into useful budgets, analytics, predictions, and savings guidance. Development is incremental; features are documented as they are implemented and verified.

## Current status

The Flask foundation, User Authentication, Transaction Management, Budget Management, Savings Goals, Advanced Financial Analytics, monthly expense prediction, transaction categorization, expense forecasting, ML model persistence/training management, and protected ML dashboard are implemented. Users can register, sign in, manage their own transactions and budgets, record goal contributions, and explore user-scoped analytics and ML predictions. CSV importing and PDF reports are not implemented; reports and deployment remain future phases. PostgreSQL remains the configured default; automated tests use in-memory SQLite.

## Architecture

- `app/__init__.py`: application factory and extension/blueprint registration.
- `app/extensions.py`: shared SQLAlchemy, migration, login, and CSRF extension instances.
- `app/models/core.py`: initial relational schema. Money uses `NUMERIC(12, 2)`; ownership is tied to users; delete cascades apply to user-owned records.
- `app/forms.py`: WTForms input validation for registration and login.
- `app/routes/web.py`: registration, login, logout, landing, and protected dashboard routes.
- `app/routes/transactions.py`: user-scoped transaction creation, listing, filters, edits, and POST deletion.
- `app/services/transactions.py`: default category initialization during account/login POST flows, transaction operations, Decimal summaries, and recent records. Category reads are read-only.
- `app/routes/budgets.py` and `app/services/budgets.py`: budget CRUD, utilization, threshold alerts, and summary calculations.
- `app/routes/savings.py` and `app/services/savings.py`: savings goals, contribution CRUD, progress/status calculation, and summary totals.
- `app/routes/analytics.py` and `app/services/analytics.py`: authenticated date-filtered analysis, spending patterns, derived insights, budget-overrun patterns, and anomaly detection.
- `app/routes/ml.py`: protected ML dashboard, prediction and forecast JSON reads, explicit training, and transaction category suggestion review actions.
- `app/templates/analytics/index.html` and `app/static/js/analytics.js`: responsive analytics dashboard and Chart.js visualizations.
- `app/services/ml/transaction_categorization.py`: per-user TF-IDF categorization pipelines, stratified evaluation, confidence-ranked suggestions, and insufficient-data handling.
- `app/services/ml/data_preprocessing.py`: user-scoped transaction preparation, exact-decimal daily/monthly expense totals, and date-derived features.
- `app/services/ml/expense_prediction.py`: next-month expense estimate, candidate regression models, previous-month baseline, and chronological evaluation.
- `app/services/ml/forecasting.py`: one-month forecast and guarded three-month flat-hold scenario built on Stage 3 predictions.
- `app/services/ml/model_store.py`: private per-user artifact storage, integrity-checked loading, training freshness, explicit refresh, and prediction history using the existing table.
- `app/templates/` and `app/static/`: responsive Bootstrap pages plus transaction form and Chart.js behavior.
- `tests/test_auth.py`, `tests/test_transactions.py`, `tests/test_budgets_savings.py`, `tests/test_analytics.py`, and ML test modules: isolated auth, CRUD, analytics, model preparation, chronological evaluation, categorization, and isolation coverage.
- `config.py`: development, testing, and production configuration.
- `migrations/finwise/`: Flask-Migrate environment, initial schema revision, and additive transaction notes revision.
- `run.py`: Flask CLI application entry point.

## Requirements and setup

Use **CPython 3.14.3** as the project target. The repository records this in `.python-version`. Flask and its extensions are Python packages without CPython-version-specific compiled wheels. Psycopg 3 supports Python 3.14; the `binary` extra is requested for convenient local setup. NumPy, Pandas, SciPy (a scikit-learn dependency), and scikit-learn provide Python 3.14 wheels in their current supported release lines. A PostgreSQL server and its database must be available for normal development.

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version  # should report Python 3.14.3
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
Copy-Item .env.example .env
```

Always run project commands after activating `.venv`; alternatively invoke `\.venv\Scripts\python.exe -m pip ...` and `\.venv\Scripts\python.exe -m flask ...` directly. Set a unique `SECRET_KEY` and the PostgreSQL `DATABASE_URL` in `.env`. Create the database and role in PostgreSQL using your local administrator setup. Then:

```powershell
$env:FLASK_APP = "run.py"
.\.venv\Scripts\python.exe -m flask db upgrade -d migrations\finwise
.\.venv\Scripts\python.exe -m flask run
```

The migration repository lives in `migrations/finwise`; its chain is `3c8978a0c365` (initial schema) → `01ac73546bac` (transaction notes) → `23eebe22d66f` (budget/savings descriptions and alert key). Apply committed migrations with `db upgrade`; do not initialize or reset migration history. In this Stage 8 environment, the configured `.env` target was identified as `localhost:5432/finwise_db`, but TCP port 5432 was unavailable. Therefore the live database's current Alembic revision, schema, PostgreSQL version, and data behavior could not be checked. PostgreSQL dialect SQL compilation did pass for all 10 ORM tables and the JSON alert predicate, but this is not a live integration test. The app health endpoint runs `SELECT 1` when the service is available. No database command was run against `finwise_db` during this audit.

## Authentication

The registration page is `/register`; it requires a full name, valid email, password of 12–128 characters, and matching confirmation. Email is normalized to lowercase, and the database unique constraint handles duplicate registrations, including concurrent attempts. Passwords are hashed with Werkzeug's password hashing implementation and are never returned to templates. Sign in at `/login`; invalid credentials receive the same generic response. `/dashboard` requires a Flask-Login session. Sign out with the CSRF-protected POST form in the navigation bar. Sessions use HTTP-only, SameSite=Lax cookies, a 12-hour lifetime, and Flask-Login's strong session protection. Production always enables secure cookies; local development may use HTTP.

For local development, activate the virtual environment and start the app with `python -m flask --app run run`. For tests, install `requirements-dev.txt` and run `python -m pytest`. `TestingConfig` uses SQLite in memory and disables CSRF only for tests; normal app configurations keep CSRF protection enabled. Authentication uses the existing `users` table, so it adds no migration.

Budget alerts and default categories are initialized by account/login or relevant POST operations; normal GET pages do not write financial records. Alembic connection URLs mask database passwords in its configuration/logging path.

## Python 3.14 compatibility

The dependency ranges include Flask 3.1, Flask-SQLAlchemy 3.1, Flask-Migrate 4.1, psycopg 3, NumPy 2.4+, Pandas 2.3+, and scikit-learn 1.8+, which support CPython 3.14 in current releases. Pandas, NumPy, SciPy, and scikit-learn contain compiled extensions, so use releases with CPython 3.14 wheels; avoid forcing old exact pins. No required package is currently known to lack a Python 3.14-compatible route. If a platform has no wheel for a particular release, prefer a newer compatible release or a platform with published wheels; building scientific packages from source on Windows may require a compiler toolchain. The project `.venv` has been installed and checked on Python 3.14.3; `pip check` reports no broken requirements.

## Budgets and savings goals

Budgets belong to one user and one of that user's expense categories. Monthly periods normalize to the full calendar month containing the selected start date; custom periods use inclusive start/end dates. Overlapping budgets for the same category are rejected. Actual spending is the sum of that user's expense transactions in the category and budget date range.

- Remaining budget = allocated amount − actual expenses.
- Usage percentage = actual expenses ÷ allocated amount × 100.
- Status is normal below 70%, warning from 70% to below 90%, critical from 90% through 100%, and overspent above 100%.
- Threshold alerts at 70%, 90%, and overspend are stored in the existing `recommendations` table. A unique `dedupe_key` prevents duplicate alerts per budget threshold; the existing `dismissed_at` field supports dismissal.

Savings goals belong to a user and own their contribution records. Contributions are the source of truth for amounts saved; saved totals are never separately stored. Remaining amount is `max(target − contributions, 0)`, and progress is capped at 100%. A goal is completed once contributions reach its target; an incomplete goal with a past target date is overdue. Editing or deleting contributions recalculates the goal completion timestamp.

Dashboard budget/savings totals and charts are queried from the authenticated user's records. Active budget totals include only budgets covering the current date. Goal totals include all of the user's goals; completed and overdue counts are shown separately through statuses.

## Advanced financial analytics

`GET /analytics` requires authentication. It defaults to the trailing 365 days and accepts `start_date` and `end_date` (`YYYY-MM-DD`) query parameters. The optional `period` parameter selects monthly or yearly chart grouping. The page reports income, expenses, net cash flow, savings rate (undefined when income is zero), daily/monthly expense averages, spending by category and period, weekday/weekend patterns, recurring-like expenses, complete-month changes, large transactions, repeated budget overruns, and derived financial insights. Results are calculated from the signed-in user's own transactions, categories, and budgets; no separate analytics or prediction records are saved.

Unusual transactions are review prompts, not fraud findings. The statistical method analyzes expense amounts only after at least 8 records, using the IQR upper fence and a 3-standard-deviation signal. Isolation Forest analyzes expense amount, category, and weekday features when at least 20 records are available. The dashboard states when the history is too small; thresholds and model flags are descriptive heuristics, not financial advice. Analytics allows up to a 10-year date range. No schema changes were needed, so this milestone adds no Alembic migration and preserves the existing migration history and PostgreSQL tables.

## Monthly expense prediction (backend service)

The Stage 3 service is `app.services.ml.expense_prediction.predict_next_month`. Callers pass the output of `fetch_user_transactions(current_user.id)` from Stage 2, so the prediction data originates from an explicitly user-scoped query. The service re-cleans its input and aggregates **expenses only** by calendar month using exact `Decimal` sums; missing calendar months inside the observed range are zero-filled. Amounts are converted to floating point only for scikit-learn model fitting, and output predictions are clamped to zero or higher and rounded to currency cents.

For each target month, features include the previous one, two, and three monthly totals; the mean of the previous three months; and cyclic sine/cosine encodings of the target month number. Lag and rolling features use only earlier periods. Candidate estimators are Linear Regression, a shallow Random Forest Regressor, and a shallow Gradient Boosting Regressor, each with deterministic tree-model random states and median imputation. The naive baseline predicts the previous month's expenses.

Evaluation uses expanding-window walk-forward validation: for each test month, candidates are fitted only on earlier supervised months. No shuffled split is used. MAE is the average absolute prediction error, RMSE penalizes larger errors more heavily, and R² is reported only when at least three test observations have non-constant actual values. A model is selected only when its validation MAE improves on the baseline by at least 5%; otherwise the baseline is preferred. Model comparison requires at least 12 calendar months of expense history. With shorter history the service returns an explicit insufficient-data status and, if any expense month exists, a clearly marked previous-month estimate without claiming model metrics. Empty or income-only history yields no prediction.

Stage 6 adds optional persistence around this prediction engine; Stage 3 itself remains independent of routes and storage. Only a selected fitted regression pipeline is saved. Previous-month baseline and insufficient-data decisions are cached as signed metadata without a model artifact. Prediction history uses the existing `predictions` table. Test data is synthetic and used only by unit tests; no real user financial data was trained or evaluated. The estimates are historical-pattern estimates, not guarantees or financial advice.

## Automatic transaction categorization (backend service)

The Stage 4 service is `app.services.ml.transaction_categorization.train_categorizer` and `predict_category`. Pass only the prepared rows returned by `fetch_user_transactions(current_user.id)`; the service rejects mixed-owner frames when a `user_id` column is present and never queries or combines other users' records. It reuses Stage 2's `prepare_labeled_transactions`, normalizes descriptions to lowercase words with punctuation and repeated whitespace removed, and treats category name plus transaction type as the class so same-named income and expense categories remain distinct.

The model pipeline uses TF-IDF (term frequency weighted by inverse document frequency) with one- and two-word features, then compares Multinomial Naive Bayes and Logistic Regression. Each category needs at least three usable labeled descriptions to enter final training; lower-support categories are excluded and reported. A deterministic stratified holdout is used only when every eligible category has at least ten examples. TF-IDF is fitted inside each scikit-learn pipeline on the training split only, then final candidate pipelines are fitted on all eligible rows. Accuracy, macro and weighted precision/recall/F1, and a confusion matrix are reported only for the held-out evaluation. Macro F1 chooses the evaluated model. If evaluation support is insufficient but at least two eligible classes remain, Logistic Regression is exposed only as an explicitly unvalidated fallback, with no evaluation metrics claimed.

Predictions include the category, transaction type, model confidence, and ranked alternatives. Confidence is a classifier probability score, not a guarantee or calibrated likelihood of correctness. Empty, very short, and entirely unseen descriptions receive no category suggestion. Stage 7 provides a protected review workflow; suggestions never update transactions until the user accepts one. Synthetic examples are test fixtures only; no real financial records were used for evaluation.

## Expense forecasting (backend service)

The Stage 5 functions are `forecast_next_month`, `forecast_three_months`, and `forecast_expenses` in `app.services.ml.forecasting`. Forecasting presents historical monthly expense actuals alongside estimates across a specified horizon. Stage 3's `predict_next_month` remains the sole prediction engine: it supplies the next-month amount, selected model or previous-month baseline, and one-month chronological evaluation metrics. This layer does not duplicate aggregation or train another estimator.

The next-month forecast can return Stage 3's clearly marked previous-month baseline when model comparison history is insufficient; the result remains marked insufficient. A three-month forecast is withheld unless at least 24 calendar months of actual expense history exist and Stage 3 reports its one-month result as sufficient. The first future month uses Stage 3. Months two and three repeat that first estimate as a flat-hold scenario, labeled separately and without evaluation metrics. Forecast outputs are never appended to actual history or passed back into Stage 3, avoiding recursive error accumulation. Stage 6's forecast wrapper uses the current saved Stage 3 model/decision and does not persist a separate forecasting model. The 24-month threshold is a conservative data sufficiency rule, not proof of forecast reliability. Multi-month accuracy is not measured by Stage 3's one-step metrics. Amounts remain non-negative and at Stage 3's cent precision. Stage 7 presents these forecasts in the protected ML dashboard.

## ML model persistence and training management

Stage 6 is implemented in `app.services.ml.model_store`. It persists only fitted per-user expense regression and transaction categorization pipelines. Forecasting has no independently trained model and reuses the expense prediction service. A selected fitted estimator is saved; simple previous-month baselines and insufficient-data outcomes are signed metadata decisions and never create Joblib artifacts. Models trained on one user's records are never shared with another user.

Artifacts and their JSON metadata are stored beneath the Flask private `instance/ml_models/user_<id>/` directory by default. The existing `.gitignore` excludes `instance/`; files are not served as static assets. Storage paths derive from a positive integer owner ID and an allowlisted model type, never a client-supplied filename. Loading validates the owner, model type/version, fixed internal artifact location, feature configuration, HMAC signature, and SHA-256 artifact digest before Joblib deserialization. The accepted pipeline classes and structure are allowlisted; user-uploaded artifacts are never loaded. `SECRET_KEY` must remain stable and private because it signs metadata. File permission mode changes are best-effort on Windows; the directory inherits the operating system's ACLs and should remain accessible only to the application account.

Metadata records the model type/version, training time and row count, approved feature configuration, training date range, evaluation metrics, owner, internal artifact location, status, and a SHA-256 fingerprint of the cleaned source transactions. It does not store raw descriptions or transaction records. A changed, added, removed, or edited source record changes the fingerprint, so the model is reported stale. Categorization changes are also fingerprinted. Prediction and forecast requests do not train implicitly. Call `train_user_model(user_id, model_type, user_scoped_data)` explicitly to train missing/stale models; unchanged current models are reused. Use `force_refresh=True` for an intentional retrain when source rows have not changed. Transaction create/edit/delete operations need no model-specific database hook: the next freshness check detects data changes.

`forecast_expenses_for_user` supplies the persisted/current Stage 3 prediction result to the Stage 5 forecasting layer. `save_prediction_history` writes predicted periods to the existing `predictions` table and upserts by owner and period; historical actual rows are never written as predictions. `user_prediction_history` scopes reads by `user_id`. The existing schema supports these fields, so this stage adds no model or prediction migration. No scheduler or background worker is introduced; callers must explicitly train/refresh after stale status is detected. Concurrent refresh coordination is process-local, so deployments with multiple workers should serialize refresh jobs at the application layer before enabling concurrent training. Changing `SECRET_KEY` invalidates existing metadata signatures, requiring explicit retraining. Stage 7 exposes these services through protected routes.

## ML dashboard and review workflow

The protected dashboard is `/ml/`. It shows the next-month prediction, model/baseline status, available evaluation metrics and training metadata, one- and (when enough history exists) three-month forecasts, an actual-versus-saved-prediction chart, category suggestions, and unusual-spending flags from the existing analytics service. `GET /ml/prediction` and `GET /ml/forecast?horizon=1|3` return user-scoped JSON results. The model uses the user's full valid history; date-range filtering is not exposed because Stage 3's prediction interface does not support date bounds.

Use the dashboard's CSRF-protected `POST /ml/train` form to explicitly train or refresh either the expense prediction model or categorizer. Stage 6 reuses a current model and only retrains if its version or source-data fingerprint is stale; force retraining is available as an explicit option. When explicit expense training produces an estimate, it is recorded in the existing prediction history table for the actual-versus-predicted chart. Dashboard reads do not train or write prediction history. Browser errors are generic and do not reveal artifact paths or exception details.

Category suggestions are requested for an owned transaction and displayed with the suggested category, model confidence score, and alternatives. The user must explicitly accept or reject. Acceptance rechecks transaction ownership, looks up the submitted category in the database, allows only the user's own or an explicitly system-owned category, verifies its transaction type, and requires it to be among the model's alternatives. Rejecting only dismisses the suggestion. Neither action trains or changes the model; accepting updates the transaction category and naturally makes its source fingerprint stale for later training. Transaction descriptions are not rendered on the ML dashboard. Chart.js overlays actual monthly totals and saved prediction-history values when available. Anomaly cards reuse analytics calculations and omit descriptions. All changes use the existing tables; Stage 7 adds no migration.

Run route tests with `python -m pytest tests/test_ml_routes.py -q`; run the complete suite with `python -m pytest -q` from the active Python 3.14 `.venv`.

## Initial database schema

`users` owns transactions, budgets, savings goals, predictions, recommendations, chat history, and imported statement records. Categories can be global (`user_id` null) or user-specific. Transactions and budgets refer to categories; savings goals own contributions. Amounts are fixed precision decimal values. Budget uniqueness is enforced per user, category, and period start. All personal data models carry a user ownership foreign key.

## API and feature status

Authentication and transaction management are currently server-rendered web routes:

- `GET /transactions`: paginated history with description, type, category, date, and amount filters.
- `GET, POST /transactions/new`: create income or expense.
- `GET, POST /transactions/<id>/edit`: edit an owned transaction.
- `POST /transactions/<id>/delete`: CSRF-protected deletion.
- `GET /dashboard`: authenticated financial overview and recent transactions.
- `GET, POST /register` and `/login`; `POST /logout`.
- `GET /budgets`: budget overview and active alerts.
- `GET, POST /budgets/new` and `/budgets/<id>/edit`; `POST /budgets/<id>/delete`.
- `POST /budgets/alerts/<id>/dismiss`.
- `GET /savings`: goal overview.
- `GET, POST /savings/new` and `/savings/<id>/edit`; `POST /savings/<id>/delete`.
- `GET, POST /savings/<goal_id>/contributions`.
- `GET, POST /savings/<goal_id>/contributions/<contribution_id>/edit`; `POST /savings/<goal_id>/contributions/<contribution_id>/delete`.
- `GET /analytics`: protected, user-specific financial analytics with a bounded date filter.
- `GET /ml/`: protected machine-learning dashboard with model status, expense predictions/forecasts, actual-versus-predicted chart, categorization review, and anomaly summaries.
- `GET /ml/prediction` and `GET /ml/forecast?horizon=1|3`: protected JSON reads for the authenticated user's saved-model result.
- `POST /ml/train`: CSRF-protected explicit training/refresh for the selected model type.
- `POST /ml/transactions/<id>/suggest`, `/accept`, and `/reject`: CSRF-protected user-owned category suggestion workflow.

The existing `users` table owns its `transactions`; every transaction references a `categories` row. Default categories are created per user and protected by the existing `(user_id, name, transaction_type)` uniqueness constraint. Transaction amounts are stored as `NUMERIC(12, 2)`. A nullable `notes` column was added by Alembic revision `01ac73546bac` without changing the existing transaction data or prior migration revisions.

Run the complete test suite with `python -m pytest -q` while `.venv` is active. The suite uses the in-memory SQLite `TestingConfig`, never the configured PostgreSQL database. It verifies route authentication and CSRF, complete transaction-budget-savings-analytics-ML workflows, date bounds, user isolation, aggregation, empty history, anomaly methods, chronological expense evaluation, TF-IDF leakage prevention, category support thresholds, confidence output, category suggestion acceptance/rejection and category validation, forecast horizons and labels, private artifact handling, freshness and explicit refresh, and prediction history ownership. These ML tests use synthetic fixtures; Stages 6 and 7 reuse existing tables and introduce no database objects or migrations.

## ML methodology, results, screenshots, and deployment

The expense prediction and categorization methodologies are described above. ML evaluation in the automated suite uses synthetic test fixtures and is not evidence of real-user performance. No production models have been trained or persisted, and no deployment has been performed. Financial time-series evaluation uses chronological validation; text categorization uses a stratified holdout when class support permits. Stage 8's automated suite passed on SQLite, but PostgreSQL integration remains unverified because the configured local server port was unavailable. Before deployment, verify the migration revision and schema on a reachable PostgreSQL 18 service, configure restricted filesystem ACLs for local model artifacts, use HTTPS with secure cookies, and account for the model store's process-local refresh lock when running multiple workers. Bootstrap and Chart.js are loaded from public CDNs, so the frontend depends on their availability unless assets are self-hosted.

## Stage 9: synthetic financial validation

The checked-in generator creates deterministic fictional INR records using seed `20260917`: 481 transactions across three `.example.test` users and 60 savings contributions. Two ledgers span 24 calendar months and cover salary/stipend, freelance income, recurring and discretionary categories, and unusual but legitimate one-time purchases. A third 12-month scenario has monthly income ₹50,000, fixed expenses ₹20,000, variable expenses ₹10,000, a ₹60,000 savings goal, and ₹5,000 monthly contributions. The fixtures contain no credentials, bank identifiers, or user-provided personal data.

Generate or safely regenerate only the three known fixture files with:

```powershell
.\.venv\Scripts\python.exe scripts\generate_synthetic_financial_data.py --environment test
# To replace those generated fixture files explicitly:
.\.venv\Scripts\python.exe scripts\generate_synthetic_financial_data.py --environment test --replace-existing
```

The generator refuses to run when `FLASK_CONFIG=production`, requires the explicit test/development flag, refuses existing targets unless `--replace-existing` is passed, and writes only to `tests/fixtures/`. It does not load `.env`, open a database connection, or write to PostgreSQL. Pytest loads these records into the existing `TestingConfig` in-memory SQLite database and drops its schema after each test. The existing tables and Alembic history are unchanged.

Stage 9 independently reconciles Decimal totals, monthly and category spending, inclusive period boundaries, budget usage/remaining amount and threshold alert deduplication, contribution history and goal progress, and INR formatting. The exact 12-month scenario reconciles ₹600,000 income, ₹360,000 expenses, ₹240,000 net balance, and ₹60,000 contributions. The suite also checks empty, income-only, expense-only, large-amount, first/last-day, and invalid zero-amount cases. SQLite is the only database exercised by these tests.

On this fixed synthetic primary ledger, chronological expense evaluation used 15 expanding-window training observations and six held-out months (July–December 2025). The previous-month baseline scored MAE ₹31,151.23, RMSE ₹46,387.17, R² −1.7323. Linear Regression scored MAE ₹24,013.31, RMSE ₹31,024.19, R² −0.2222; Random Forest scored ₹19,216.46, ₹29,108.49, −0.0759; Gradient Boosting scored ₹19,079.88, ₹28,003.02, 0.0043. Gradient Boosting was selected by the existing 5% MAE improvement rule. R² remains near zero, so the synthetic monthly totals are hard to explain with the available lag/calendar features. These values are fixture-level checks, not expected production scores. Predictions are non-negative, and Stage 5's three-month view labels estimates separately from recorded actuals.

Categorization was trained only through June 2025 and evaluated on 58 eligible labeled rows from July–December 2025. The external chronological results were accuracy 0.9828, macro precision 0.9881, macro recall 0.9722, and macro F1 0.9769. Empty, short, unfamiliar, multi-class, income/expense, confidence, and insufficient-class-support outcomes are checked; this one synthetic split does not establish real-world accuracy.

The anomaly review flags all three deliberately injected large/spike expenses, and after correction its category-level statistical method no longer flags any of the 24 recurring ₹18,000 rent payments. The combined detector still returned 10 review candidates: Isolation Forest also flagged seven records not marked as injected anomalies in this fixture (including low-support travel and bank-charge records). These are candidates for human review, not fraud findings. The statistical method only evaluates categories with at least eight observations; sparse categories rely on the Isolation Forest and therefore have greater false-positive/false-negative uncertainty.

Dashboard and analytics routes were exercised through Flask's test client using the same seeded database; chart payloads, INR output, explicit model training, saved actual-versus-predicted history, forecast labels, and empty-history behavior were checked. No interactive browser session was used. PostgreSQL remains pending live integration verification because the configured local port was unavailable in the prior environment check; no command in Stage 9 connected to or changed `finwise_db`.

The Stage 9 complete suite passed 132 tests using the Python 3.14 `.venv`. Run all verification commands from the project root with that environment:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m compileall app tests
.\.venv\Scripts\python.exe -m pip check
```

## Stage 10: Render deployment preparation

This repository is prepared for a manual Render deployment. No service or database was created and no migration was run. The project targets CPython 3.14.3 in `.python-version`; Render's native Python runtime supports released Python 3.14 patch versions ([Render Python version guide](https://render.com/docs/python-version)). Gunicorn is included in `requirements.txt`; its current PyPI release is a universal Python wheel and supports Python 3.10+ ([Gunicorn on PyPI](https://pypi.org/project/gunicorn/)). Local development continues to use Flask's development server, not Gunicorn.

### GitHub and Render setup

The current workspace has no Git repository metadata. Before using Render, initialize or clone the project into your own GitHub repository, review the changes, and push the branch you intend to deploy. `.gitignore` excludes `.env`, other `.env.*` files except the safe `.env.example`, `instance/`, local SQLite files, uploaded files, and model artifacts. Confirm `.env` is ignored and is not staged; never commit actual credentials or generated model files.

Create a Render account and manually create a managed PostgreSQL database and Python web service from the GitHub repository. For a service primarily serving India, Singapore is the closest currently listed Render region; choose the same region for the web service and database so the service can use the database's private connection string. Region selection cannot be changed after resource creation, so verify it before creating either resource ([Render regions](https://render.com/docs/regions)).

No `render.yaml` is included. Render assigns a compute plan when a Blueprint omits one; current documented defaults are a `0.5c-512mb` web service and `0.1c-256mb` Postgres database. Selecting either default would silently make a paid/free-plan decision for you. Choose the service and database plans in Render before creating resources, then copy the settings below into the dashboard. Render's free Postgres currently expires after 30 days and is not suitable for production data ([Render Blueprint defaults](https://render.com/docs/blueprint-spec), [Render free instance limitations](https://render.com/docs/free)).

### Build and start settings

Use these settings for the web service:

- Runtime: Python. Render reads the root `.python-version` file (`3.14.3`); do not set an older override.
- Region: Singapore is a suggested starting point for India-based users; keep Postgres in the same region.
- Build command: `pip install -r requirements.txt`
- Start command: `gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120 run:app`
- Health check path: `/health`
- Set `FLASK_CONFIG=production`.

Render requires a public web service to bind to `0.0.0.0` and recommends using its `PORT` variable. Render terminates HTTPS at its edge and forwards traffic to the service over HTTP; the production app trusts only one `X-Forwarded-Proto` hop so secure-cookie and HTTPS URL handling work ([Render web services and TLS](https://render.com/docs/web-services)).

Set these variables in the Render service dashboard:

| Variable | Value |
| --- | --- |
| `FLASK_CONFIG` | `production` |
| `SECRET_KEY` | A unique random value of at least 32 bytes. Generate locally with `python -c "import secrets; print(secrets.token_urlsafe(48))"` and save it only in Render's secret environment settings. |
| `DATABASE_URL` | The managed database's **internal** connection string. The app converts `postgres://` and `postgresql://` to the installed `postgresql+psycopg://` driver form. |
| `TRUSTED_HOSTS` | Optional comma-separated hostnames for custom domains. Render's `RENDER_EXTERNAL_HOSTNAME` is added automatically when present. Do not include schemes or paths. |
| `ML_MODEL_STORE_DIR` | Optional. Set only when a persistent disk is attached, e.g. `/var/data/ml_models`; otherwise leave unset. |

Do not set `SESSION_COOKIE_SECURE=false` in production. Production fixes secure and HTTPOnly cookies on, uses SameSite `Lax`, keeps CSRF enabled, and requires the secret and database URL from the process environment. Development and test configuration remain separate. Never copy `.env` values into Render or the repository.

### Database migrations

After creating the hosted database and setting its internal `DATABASE_URL`, review the committed Alembic history and take a backup if the database contains data. Apply migrations through Flask-Migrate from the Render service shell:

```sh
python -m flask --app run db history
python -m flask --app run db current
python -m flask --app run db upgrade
python -m flask --app run db current
```

Run `db upgrade` only after confirming the Render environment is connected to the intended new hosted database. This project does not run `db.create_all()` in production, and no local or hosted migration was run during preparation. Do not run `db init`, `db stamp`, `db downgrade`, or destructive SQL to prepare this deployment. The existing migration chain remains the source of truth. Keep the service unavailable to end users until the migration completes and `/health` returns success.

### Model artifacts and filesystem persistence

Render's service filesystem is ephemeral by default; writes outside a mounted persistent disk are lost on redeploy/restart ([Render persistent disks](https://render.com/docs/disks), [deploy filesystem behavior](https://render.com/docs/deploys)). FinWise stores signed per-user Joblib artifacts privately under `instance/ml_models` by default. Missing or stale artifacts fail closed; the app reports the model as unavailable and requires explicit training from the protected ML workflow. Dashboard reads do not train models.

For initial deployment, leave `ML_MODEL_STORE_DIR` unset and expect artifacts to disappear when the service restarts. Users can explicitly train again after a missing-artifact status. For artifacts that must survive restarts, select a paid web-service plan, attach a Render persistent disk mounted at `/var/data`, then set `ML_MODEL_STORE_DIR=/var/data/ml_models`. Persistent disks add cost, only preserve files under the mount, and disable Render's zero-downtime deploy behavior. This file-backed store is not shared between multiple service instances; keep one instance unless you later adopt a shared artifact store or another suitable design. No cloud object storage was added.

### Health, errors, and logging

`GET /health` executes only `SELECT 1`. It returns `200` with a minimal status when PostgreSQL is reachable and `503` with a generic unavailable response otherwise. It does not run analytics, predictions, or model training. Production error responses are generic; unexpected exceptions are logged as a static message without traceback or exception text. Application routes do not log financial descriptions, credentials, session cookies, or database URLs. Alembic already masks passwords when rendering database URLs.

### Troubleshooting and post-deployment checks

- Startup says `SECRET_KEY`: set a random Render secret with at least 32 bytes and redeploy.
- Startup rejects `DATABASE_URL`: copy the database's internal connection string; confirm the database and service share a region. The app accepts standard Render PostgreSQL schemes and selects Psycopg 3.
- `/health` returns `503`: verify database status, internal host/port, access rules, and service region. The response intentionally omits credentials and exception details; inspect Render's protected service logs for platform-level connection failures.
- Login/forms reject requests: confirm HTTPS is enabled and CSRF tokens are present; do not disable CSRF or secure cookies.
- ML status says artifact missing/stale after restart: explicitly retrain. If persistence is required, attach the paid persistent disk and configure `ML_MODEL_STORE_DIR`; confirm it is mounted before training.
- Build fails on `gunicorn`: confirm `requirements.txt` is used as the build command and the selected runtime reads `.python-version`.
- `ModuleNotFoundError` or import failure: confirm Render is building from the repository root and using `run:app` as the WSGI target.

After the manual deployment and migration, verify `db current`, `GET /health`, registration/login/logout, CSRF-protected transaction creation, ownership isolation, budgets and savings pages, and ML's missing-artifact status. Train a model only through the explicit authenticated workflow, then verify the resulting status and persistence behavior after a planned restart. Check the service logs and database backups before sharing the public URL. The `/health` check and deployment commands have not been run against Render from this workspace.
