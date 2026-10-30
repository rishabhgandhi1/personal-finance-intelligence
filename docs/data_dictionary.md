Personal Finance Intelligence — Data Dictionary

1. Transaction CSV Format

The application accepts CSV files containing these columns:

Column	Required	Description	Example
transaction_id	Yes	Unique identifier for the transaction	TXN000001
transaction_date	Yes	Date of the transaction in YYYY-MM-DD format	2026-10-01
description	Yes	Transaction description or merchant	Grocery Store
amount	Yes	Positive monetary amount with up to two decimal places	450.00
transaction_type	Yes	INCOME or EXPENSE	EXPENSE
category	No	Existing category or blank if unknown	Groceries
payment_method	No	Payment method	UPI
account_type	No	Account type	BANK_ACCOUNT

2. Amount Convention

All transaction amounts are positive.

The transaction_type field determines whether an amount represents income or an expense.

Net cash flow = Total income - Total expenses.

3. Supported Categories

* Food
* Groceries
* Transport
* Shopping
* Utilities
* Healthcare
* Entertainment
* Education
* Housing
* Travel
* Personal Care
* Subscriptions
* Income
* Other

A blank category is allowed for transactions that require categorization.

4. Transaction Types

Supported transaction types are INCOME and EXPENSE.

Values are normalized to uppercase during ingestion.

5. Date Format

Dates must be valid calendar dates in YYYY-MM-DD format.

6. Validation Rules

* Required columns must exist.
* Transaction IDs must not be blank.
* Transaction IDs must be unique within a file and across previously imported transactions.
* Transaction dates must be valid.
* Amounts must be finite, positive, and contain at most two decimal places.
* Transaction types must be INCOME or EXPENSE.
* Descriptions must not be blank.
* Unsupported categories must be rejected rather than silently changed.
* Blank categories are allowed.
* Missing optional payment methods default to UNKNOWN.
* Missing optional account types default to BANK_ACCOUNT.

7. Duplicate Handling

A transaction with an ID already present in the database must not be inserted again.

Duplicate rows within the same file must also be identified.

The ingestion report will distinguish invalid rows from duplicate rows.

8. Privacy

Use synthetic transaction data for development and demonstrations.

Never commit real personal financial information or API credentials to the public repository.