# Reference Data Comparison Results

## PDF Format Results

### Row-Spanning Data Check
Found spanning data:
- {'value': '<= $1,000,000', 'span_index': 1, 'span_total': 3}
- {'value': '<= $1,000,000', 'span_index': 2, 'span_total': 3}
- {'value': '<= $1,000,000', 'span_index': 3, 'span_total': 3}


Extracted loan amounts:
- {'value': '<= $1,000,000', 'span_index': 1, 'span_total': 3}
- {'value': '<= $1,000,000', 'span_index': 2, 'span_total': 3}
- {'value': '<= $1,000,000', 'span_index': 3, 'span_total': 3}

Reference spanning data:
- <= $1,000,000 (1)
- <= $1,000,000 (2)
- <= $1,000,000 (3)
- $250,000 (1)
- $250,000 (2)
- $3,000 (1)
- $3,000 (2)

Comparing:
- Reference: <= $1,000,000 (1) (cleaned amount: $1,000,000, numeric: 1000000, span: 1)
- Extracted: <= $1,000,000 (numeric: 1000000, span: 1)
- LTE match: True, Dollar match: True

Match found!
### Geographic Restrictions
Found:
- NJ - 10% reduction
- CT/IL - 5% reduction

Expected:
### Reserve Requirements
Found: 12 months PITI required for loan amounts > $1,000,000
Expected:
- 6 Months

- 12 Months (1)

- 12 Months (2)

- 12 Months (3)

- 6 Months PITIA from departing residence (1)

- 48 Months (1)

- 6 Months PITIA from departing residence (2)

- 48 Months (2)

- 48 Months (3)

- 48 Months (4)

### Heading Structure Check
Found main headings:
- Program Max LTVs
- Requirements

Expected main headings:
- Program Max LTVs
- Requirements


Detailed Property/Transaction Types:
- primary_residence ^ purchase
- primary_residence ^ rate_and_term
- primary_residence ^ cash_out

Reference Property/Transaction Types:
- Program Max LTVs ^ Loan Amount
- Program Max LTVs ^ FICO
- Requirements ^ Program Requirements
- Requirements ^ Property Overlays
- Requirements ^ Max LTV
- Requirements ^ Products
- Requirements ^ ARM Margins & Caps
- Requirements ^ Max DTI
- Requirements ^ Interest Only
- Requirements ^ Qualifying Rate
- Requirements ^ Reserve Requirements
- Requirements ^ Credit Requirements
- Requirements ^ Qualifying Payment
## PNG Format Results

### Row-Spanning Data Check
Found spanning data:
- {'value': '<= $1,000,000', 'span_index': 1, 'span_total': 3}
- {'value': '<= $1,000,000', 'span_index': 2, 'span_total': 3}
- {'value': '<= $1,000,000', 'span_index': 3, 'span_total': 3}


Extracted loan amounts:
- {'value': '<= $1,000,000', 'span_index': 1, 'span_total': 3}
- {'value': '<= $1,000,000', 'span_index': 2, 'span_total': 3}
- {'value': '<= $1,000,000', 'span_index': 3, 'span_total': 3}

Reference spanning data:
- <= $1,000,000 (1)
- <= $1,000,000 (2)
- <= $1,000,000 (3)
- $250,000 (1)
- $250,000 (2)
- $3,000 (1)
- $3,000 (2)

Comparing:
- Reference: <= $1,000,000 (1) (cleaned amount: $1,000,000, numeric: 1000000, span: 1)
- Extracted: <= $1,000,000 (numeric: 1000000, span: 1)
- LTE match: True, Dollar match: True

Match found!
### Geographic Restrictions
Found:
- NJ - 10% reduction
- CT/IL - 5% reduction

Expected:
### Reserve Requirements
Found: 12 months PITI required for loan amounts > $1,000,000
Expected:
- 6 Months

- 12 Months (1)

- 12 Months (2)

- 12 Months (3)

- 6 Months PITIA from departing residence (1)

- 48 Months (1)

- 6 Months PITIA from departing residence (2)

- 48 Months (2)

- 48 Months (3)

- 48 Months (4)

### Heading Structure Check
Found main headings:
- Program Max LTVs
- Requirements

Expected main headings:
- Program Max LTVs
- Requirements


Detailed Property/Transaction Types:
- primary_residence ^ purchase
- primary_residence ^ rate_and_term
- primary_residence ^ cash_out

Reference Property/Transaction Types:
- Program Max LTVs ^ Loan Amount
- Program Max LTVs ^ FICO
- Requirements ^ Program Requirements
- Requirements ^ Property Overlays
- Requirements ^ Max LTV
- Requirements ^ Products
- Requirements ^ ARM Margins & Caps
- Requirements ^ Max DTI
- Requirements ^ Interest Only
- Requirements ^ Qualifying Rate
- Requirements ^ Reserve Requirements
- Requirements ^ Credit Requirements
- Requirements ^ Qualifying Payment
