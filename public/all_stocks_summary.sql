create view all_stocks_summary
            ("Ticker", "Name", "Region", "Sector", "Industry", "Country", "Market Cap", "Last Price", "P/E (LTM)",
             "EV/EBITDA (LTM)", "Div Yield (LTM)", "Total Return (YTD)", "Beta (1Y)", "Analyst Rating", "Last Updated")
as
-- missing source code
;

comment on view all_stocks_summary is 'Simplified view of all_stocks table with most commonly queried columns';

alter table all_stocks_summary
    owner to postgres;

