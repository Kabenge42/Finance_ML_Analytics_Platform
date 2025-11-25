# Phase 9.3 New Columns Mapping

## Summary

48 new columns to add to SQL schemas and data.py schema_mapping dictionary

## Column Mappings (SQL Name → Normalized Name → Data Type)

### Category 1: Revenue Forecasting Estimates (4 columns)

1. "Revenues - Est Avg (NTM)" → "revenues_est_avg_ntm" → NUMERIC
2. "Revenues - Est Avg (FY1E)" → "revenues_est_avg_fy1e" → NUMERIC
3. "Revenues - Est Med (NTM)" → "revenues_est_med_ntm" → NUMERIC
4. "Revenues - Est Med (FY1E)" → "revenues_est_med_fy1e" → NUMERIC

### Category 2: EV/Sales Time-Series (11 columns)

5. "EV/Sales (EST FY1)" → "ev_sales_est_fy1" → NUMERIC
6. "EV/Sales (LTM)" → "ev_sales_ltm" → NUMERIC
7. "EV/Sales (NTM)" → "ev_sales_ntm" → NUMERIC
8. "EV/Sales (-1FYLTM)" → "ev_sales_1fyltm" → NUMERIC
9. "EV/Sales (-2FYLTM)" → "ev_sales_2fyltm" → NUMERIC
10. "EV/Sales (-3FYLTM)" → "ev_sales_3fyltm" → NUMERIC
11. "EV/Sales (3YAVGLTM)" → "ev_sales_3yavgltm" → NUMERIC
12. "EV/Sales (-1FQLTM)" → "ev_sales_1fqltm" → NUMERIC
13. "EV/Sales (-2FQLTM)" → "ev_sales_2fqltm" → NUMERIC
14. "EV/Sales (-3FQLTM)" → "ev_sales_3fqltm" → NUMERIC
15. "EV/Sales (-4FQLTM)" → "ev_sales_4fqltm" → NUMERIC

### Category 3: Employment Metrics (2 columns)

16. "Total Employees (FY)" → "total_employees_fy" → NUMERIC
17. "Total Employees (FQ)" → "total_employees_fq" → NUMERIC

### Category 4: Technical Indicators (6 columns)

18. "52W High/Adj" → "52w_high_adj" → NUMERIC
19. "52W Low/Adj" → "52w_low_adj" → NUMERIC
20. "EMA (20D)" → "ema_20d" → NUMERIC
21. "EMA (50D)" → "ema_50d" → NUMERIC
22. "EMA (100D)" → "ema_100d" → NUMERIC
23. "EMA (250D)" → "ema_250d" → NUMERIC

### Category 5: EV/EBITDA Extended Time-Series (6 columns)

24. "EV/EBITDA (LTM)" → "ev_ebitda_ltm" → NUMERIC
25. "EV/EBITDA (NTM)" → "ev_ebitda_ntm" → NUMERIC
26. "EV/EBITDA (-1FYLTM)" → "ev_ebitda_1fyltm" → NUMERIC
27. "EV/EBITDA (-1FQLTM)" → "ev_ebitda_1fqltm" → NUMERIC
28. "EV/EBITDA (3YAVGLTM)" → "ev_ebitda_3yavgltm" → NUMERIC
29. "EV/EBITDA (EST FY1)" → "ev_ebitda_est_fy1" → NUMERIC

### Category 6: P/E Extended Time-Series (11 columns)

30. "P/E (EST FY1)" → "p_e_est_fy1" → NUMERIC
31. "P/E (-2FYLTM)" → "p_e_2fyltm" → NUMERIC
32. "P/E (-3FYLTM)" → "p_e_3fyltm" → NUMERIC
33. "P/E (3YAVGLTM)" → "p_e_3yavgltm" → NUMERIC
34. "P/E (-1FQLTM)" → "p_e_1fqltm" → NUMERIC
35. "P/E (-2FQLTM)" → "p_e_2fqltm" → NUMERIC
36. "P/E (-3FQLTM)" → "p_e_3fqltm" → NUMERIC
37. "P/E (-0FQQoQLTM)" → "p_e_0fqqoqltm" → NUMERIC
38. "P/E (-0FYYoYLTM)" → "p_e_0fyyoyltm" → NUMERIC
39. "P/E (-1FYYoYLTM)" → "p_e_1fyyoyltm" → NUMERIC
40. "P/E (-0FQYoYLTM)" → "p_e_0fqyoyltm" → NUMERIC

### Category 7: Dividend Record Information (8 columns)

41. "Dividend Record (Announce Date)" → "dividend_record_announce_date" → DATE
42. "Dividend Record (Ex Date)" → "dividend_record_ex_date" → DATE
43. "Dividend Record (Payable Date)" → "dividend_record_payable_date" → DATE
44. "Dividend Record (Record Date)" → "dividend_record_record_date" → DATE
45. "Dividend Record (Frequency)" → "dividend_record_frequency" → TEXT
46. "Dividend Record (Currency)" → "dividend_record_currency" → TEXT
47. "Dividend Record (Amount)" → "dividend_record_amount" → NUMERIC
48. "Dividend Streak" → "dividend_streak" → NUMERIC

## Implementation Notes

### SQL Schema Updates

- Add these columns to create_equities_schema.sql after line 262 (after "Marketing Expenses (5YAVGLTM)")
- Add these columns to create_equities_schema_sqlite.sql in same location
- Maintain quoted identifiers for SQL column names
- Use appropriate data types: NUMERIC for financial metrics, DATE for date columns, TEXT for categorical

### data.py schema_mapping Updates

- Add mappings to schema_mapping dictionary before line 351 (closing brace)
- Maintain alphabetical or logical grouping within categories
- Follow existing pattern: "SQL Name": "normalized_name"

### Normalization Pattern Applied

- Parentheses removed: (NTM) → _ntm
- Slashes to underscores: EV/Sales → ev_sales, 52W High/Adj → 52w_high_adj
- Negative prefix: (-1FY) → _1fy, (-2FY) → _2fy (minus sign dropped)
- Special chars removed: % → pct, # → (dropped), & → (dropped), . → (dropped)
- Spaces to underscores: all spaces → _
- All lowercase
