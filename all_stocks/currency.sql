DROP TABLE IF EXISTS currencies;
CREATE TABLE currencies
(
    "Date"  TEXT,
    D0      TEXT,
    D1      TEXT,
    "Value" DOUBLE PRECISION
);
CREATE OR REPLACE FUNCTION parse_year_month_to_end_of_month(date_text TEXT)
    RETURNS DATE AS
$$
DECLARE
    year_val  INTEGER;
    month_val INTEGER;
BEGIN
    IF date_text IS NULL OR TRIM(date_text) = '' THEN
        RETURN NULL;
    END IF;

    -- Validate format YYYY-MM
    IF date_text !~ '^\d{4}-\d{2}$' THEN
        RETURN NULL;
    END IF;

    year_val := SPLIT_PART(date_text, '-', 1)::INTEGER;
    month_val := SPLIT_PART(date_text, '-', 2)::INTEGER;

    -- Validate ranges
    IF year_val < 1900 OR year_val > 2100 OR month_val < 1 OR month_val > 12 THEN
        RETURN NULL;
    END IF;

    -- Return last day of month
    RETURN (MAKE_DATE(year_val, month_val, 1) + INTERVAL '1 month - 1 day')::DATE;
END;
$$ LANGUAGE plpgsql IMMUTABLE;


INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2020-12', 'M0', 'EUR1', '1.08156');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2020-12', 'M0', 'GBP1', '1.19369');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2020-12', 'M0', 'DKK100', '14.53502');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2020-12', 'M0', 'NOK100', '10.20189');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2020-12', 'M0', 'CZK100', '4.11161');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2020-12', 'M0', 'HUF100', '0.30116');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2020-12', 'M0', 'PLN100', '24.15173');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2020-12', 'M0', 'RUB1', '0.01198');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2020-12', 'M0', 'SEK100', '10.62872');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2020-12', 'M0', 'TRY100', '11.51838');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2020-12', 'M0', 'USD1', '0.88887');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2020-12', 'M0', 'CAD1', '0.69343');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2020-12', 'M0', 'ARS1', '0.01077');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2020-12', 'M0', 'BRL100', '17.24938');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2020-12', 'M0', 'MXN100', '4.45096');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2020-12', 'M0', 'ZAR1', '0.05962');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2020-12', 'M0', 'JPY100', '0.8564');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2020-12', 'M0', 'AUD1', '0.66876');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2020-12', 'M0', 'CNY100', '13.58844');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2020-12', 'M0', 'HKD100', '11.46601');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2020-12', 'M0', 'KRW100', '0.08125');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2020-12', 'M0', 'MYR100', '21.91408');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2020-12', 'M0', 'NZD1', '0.63016');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2020-12', 'M0', 'SGD1', '0.66687');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2020-12', 'M0', 'THB100', '2.95494');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2020-12', 'M0', 'XDR1', '1.27741');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2020-12', 'M0', 'USD3M', '0.886');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2020-12', 'M0', 'USD6M', '0.8836');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-01', 'M0', 'EUR1', '1.07931');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-01', 'M0', 'GBP1', '1.2092');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-01', 'M0', 'DKK100', '14.50958');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-01', 'M0', 'NOK100', '10.41457');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-01', 'M0', 'CZK100', '4.12933');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-01', 'M0', 'HUF100', '0.30062');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-01', 'M0', 'PLN100', '23.8027');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-01', 'M0', 'RUB1', '0.01192');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-01', 'M0', 'SEK100', '10.69221');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-01', 'M0', 'TRY100', '11.99406');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-01', 'M0', 'USD1', '0.88663');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-01', 'M0', 'CAD1', '0.69679');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-01', 'M0', 'ARS1', '0.01033');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-01', 'M0', 'BRL100', '16.57994');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-01', 'M0', 'MXN100', '4.4537');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-01', 'M0', 'ZAR1', '0.05863');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-01', 'M0', 'JPY100', '0.85465');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-01', 'M0', 'AUD1', '0.68486');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-01', 'M0', 'CNY100', '13.70628');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-01', 'M0', 'HKD100', '11.43635');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-01', 'M0', 'KRW100', '0.08068');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-01', 'M0', 'MYR100', '21.95528');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-01', 'M0', 'NZD1', '0.63784');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-01', 'M0', 'SGD1', '0.6687');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-01', 'M0', 'THB100', '2.95486');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-01', 'M0', 'XDR1', '1.27895');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-01', 'M0', 'USD3M', '0.884');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-01', 'M0', 'USD6M', '0.8817');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-02', 'M0', 'EUR1', '1.08554');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-02', 'M0', 'GBP1', '1.24399');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-02', 'M0', 'DKK100', '14.59703');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-02', 'M0', 'NOK100', '10.56249');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-02', 'M0', 'CZK100', '4.19592');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-02', 'M0', 'HUF100', '0.30307');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-02', 'M0', 'PLN100', '24.13525');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-02', 'M0', 'RUB1', '0.01207');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-02', 'M0', 'SEK100', '10.76438');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-02', 'M0', 'TRY100', '12.67175');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-02', 'M0', 'USD1', '0.89725');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-02', 'M0', 'CAD1', '0.70691');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-02', 'M0', 'ARS1', '0.01013');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-02', 'M0', 'BRL100', '16.58829');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-02', 'M0', 'MXN100', '4.4258');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-02', 'M0', 'ZAR1', '0.06081');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-02', 'M0', 'JPY100', '0.8516');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-02', 'M0', 'AUD1', '0.69602');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-02', 'M0', 'CNY100', '13.89589');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-02', 'M0', 'HKD100', '11.57295');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-02', 'M0', 'KRW100', '0.08073');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-02', 'M0', 'MYR100', '22.17867');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-02', 'M0', 'NZD1', '0.65021');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-02', 'M0', 'SGD1', '0.67603');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-02', 'M0', 'THB100', '2.99003');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-02', 'M0', 'XDR1', '1.29246');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-02', 'M0', 'USD3M', '0.895');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-02', 'M0', 'USD6M', '0.8927');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-03', 'M0', 'EUR1', '1.10624');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-03', 'M0', 'GBP1', '1.28826');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-03', 'M0', 'DKK100', '14.87645');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-03', 'M0', 'NOK100', '10.90131');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-03', 'M0', 'CZK100', '4.22369');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-03', 'M0', 'HUF100', '0.30249');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-03', 'M0', 'PLN100', '24.05762');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-03', 'M0', 'RUB1', '0.01248');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-03', 'M0', 'SEK100', '10.87807');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-03', 'M0', 'TRY100', '12.11563');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-03', 'M0', 'USD1', '0.92945');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-03', 'M0', 'CAD1', '0.73898');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-03', 'M0', 'ARS1', '0.01021');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-03', 'M0', 'BRL100', '16.47986');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-03', 'M0', 'MXN100', '4.4692');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-03', 'M0', 'ZAR1', '0.06201');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-03', 'M0', 'JPY100', '0.85509');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-03', 'M0', 'AUD1', '0.71622');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-03', 'M0', 'CNY100', '14.27819');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-03', 'M0', 'HKD100', '11.96933');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-03', 'M0', 'KRW100', '0.0822');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-03', 'M0', 'MYR100', '22.61039');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-03', 'M0', 'NZD1', '0.66308');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-03', 'M0', 'SGD1', '0.69227');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-03', 'M0', 'THB100', '3.01906');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-03', 'M0', 'XDR1', '1.3275');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-03', 'M0', 'USD3M', '0.927');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-03', 'M0', 'USD6M', '0.9246');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-04', 'M0', 'EUR1', '1.10326');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-04', 'M0', 'GBP1', '1.27488');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-04', 'M0', 'DKK100', '14.83537');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-04', 'M0', 'NOK100', '10.98858');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-04', 'M0', 'CZK100', '4.25505');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-04', 'M0', 'HUF100', '0.30578');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-04', 'M0', 'PLN100', '24.16934');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-04', 'M0', 'RUB1', '0.01211');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-04', 'M0', 'SEK100', '10.85842');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-04', 'M0', 'TRY100', '11.27475');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-04', 'M0', 'USD1', '0.92125');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-04', 'M0', 'CAD1', '0.73686');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-04', 'M0', 'ARS1', '0.00993');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-04', 'M0', 'BRL100', '16.56554');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-04', 'M0', 'MXN100', '4.59498');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-04', 'M0', 'ZAR1', '0.06395');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-04', 'M0', 'JPY100', '0.84553');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-04', 'M0', 'AUD1', '0.70973');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-04', 'M0', 'CNY100', '14.13592');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-04', 'M0', 'HKD100', '11.85796');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-04', 'M0', 'KRW100', '0.08246');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-04', 'M0', 'MYR100', '22.35603');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-04', 'M0', 'NZD1', '0.65705');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-04', 'M0', 'SGD1', '0.69067');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-04', 'M0', 'THB100', '2.93898');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-04', 'M0', 'XDR1', '1.31872');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-04', 'M0', 'USD3M', '0.9207');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-04', 'M0', 'USD6M', '0.9183');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-05', 'M0', 'EUR1', '1.09679');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-05', 'M0', 'GBP1', '1.27102');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-05', 'M0', 'DKK100', '14.74952');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-05', 'M0', 'NOK100', '10.8777');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-05', 'M0', 'CZK100', '4.28878');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-05', 'M0', 'HUF100', '0.30983');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-05', 'M0', 'PLN100', '24.2087');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-05', 'M0', 'RUB1', '0.0122');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-05', 'M0', 'SEK100', '10.81189');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-05', 'M0', 'TRY100', '10.78434');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-05', 'M0', 'USD1', '0.90282');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-05', 'M0', 'CAD1', '0.74392');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-05', 'M0', 'ARS1', '0.0096');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-05', 'M0', 'BRL100', '17.02507');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-05', 'M0', 'MXN100', '4.51997');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-05', 'M0', 'ZAR1', '0.06419');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-05', 'M0', 'JPY100', '0.82732');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-05', 'M0', 'AUD1', '0.70088');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-05', 'M0', 'CNY100', '14.0439');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-05', 'M0', 'HKD100', '11.62675');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-05', 'M0', 'KRW100', '0.08039');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-05', 'M0', 'MYR100', '21.87165');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-05', 'M0', 'NZD1', '0.65228');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-05', 'M0', 'SGD1', '0.67893');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-05', 'M0', 'THB100', '2.88732');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-05', 'M0', 'XDR1', '1.30051');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-05', 'M0', 'USD3M', '0.9006');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-05', 'M0', 'USD6M', '0.8985');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-06', 'M0', 'EUR1', '1.09398');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-06', 'M0', 'GBP1', '1.27357');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-06', 'M0', 'DKK100', '14.71126');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-06', 'M0', 'NOK100', '10.78565');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-06', 'M0', 'CZK100', '4.29922');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-06', 'M0', 'HUF100', '0.31283');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-06', 'M0', 'PLN100', '24.30604');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-06', 'M0', 'RUB1', '0.01251');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-06', 'M0', 'SEK100', '10.81675');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-06', 'M0', 'TRY100', '10.53531');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-06', 'M0', 'USD1', '0.9079');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-06', 'M0', 'CAD1', '0.74305');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-06', 'M0', 'ARS1', '0.00954');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-06', 'M0', 'BRL100', '18.02775');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-06', 'M0', 'MXN100', '4.53609');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-06', 'M0', 'ZAR1', '0.06526');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-06', 'M0', 'JPY100', '0.82468');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-06', 'M0', 'AUD1', '0.69401');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-06', 'M0', 'CNY100', '14.13304');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-06', 'M0', 'HKD100', '11.69718');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-06', 'M0', 'KRW100', '0.08092');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-06', 'M0', 'MYR100', '21.95766');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-06', 'M0', 'NZD1', '0.6457');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-06', 'M0', 'SGD1', '0.68088');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-06', 'M0', 'THB100', '2.8879');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-06', 'M0', 'XDR1', '1.30273');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-06', 'M0', 'USD3M', '0.9057');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-06', 'M0', 'USD6M', '0.9035');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-07', 'M0', 'EUR1', '1.0854');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-07', 'M0', 'GBP1', '1.26725');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-07', 'M0', 'DKK100', '14.59438');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-07', 'M0', 'NOK100', '10.45764');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-07', 'M0', 'CZK100', '4.23432');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-07', 'M0', 'HUF100', '0.30378');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-07', 'M0', 'PLN100', '23.78405');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-07', 'M0', 'RUB1', '0.01242');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-07', 'M0', 'SEK100', '10.64391');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-07', 'M0', 'TRY100', '10.66861');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-07', 'M0', 'USD1', '0.91801');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-07', 'M0', 'CAD1', '0.73309');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-07', 'M0', 'ARS1', '0.00955');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-07', 'M0', 'BRL100', '17.8162');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-07', 'M0', 'MXN100', '4.59495');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-07', 'M0', 'ZAR1', '0.06321');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-07', 'M0', 'JPY100', '0.83257');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-07', 'M0', 'AUD1', '0.68158');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-07', 'M0', 'CNY100', '14.18181');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-07', 'M0', 'HKD100', '11.81348');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-07', 'M0', 'KRW100', '0.08018');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-07', 'M0', 'MYR100', '21.84979');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-07', 'M0', 'NZD1', '0.64108');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-07', 'M0', 'SGD1', '0.67758');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-07', 'M0', 'THB100', '2.81308');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-07', 'M0', 'XDR1', '1.30625');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-07', 'M0', 'USD3M', '0.9158');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-07', 'M0', 'USD6M', '0.9132');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-08', 'M0', 'EUR1', '1.07602');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-08', 'M0', 'GBP1', '1.26153');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-08', 'M0', 'DKK100', '14.46858');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-08', 'M0', 'NOK100', '10.32178');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-08', 'M0', 'CZK100', '4.22482');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-08', 'M0', 'HUF100', '0.30582');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-08', 'M0', 'PLN100', '23.55002');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-08', 'M0', 'RUB1', '0.01243');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-08', 'M0', 'SEK100', '10.53222');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-08', 'M0', 'TRY100', '10.78453');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-08', 'M0', 'USD1', '0.91421');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-08', 'M0', 'CAD1', '0.72575');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-08', 'M0', 'ARS1', '0.00941');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-08', 'M0', 'BRL100', '17.39887');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-08', 'M0', 'MXN100', '4.55599');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-08', 'M0', 'ZAR1', '0.06186');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-08', 'M0', 'JPY100', '0.83228');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-08', 'M0', 'AUD1', '0.66746');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-08', 'M0', 'CNY100', '14.11566');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-08', 'M0', 'HKD100', '11.74486');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-08', 'M0', 'KRW100', '0.07874');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-08', 'M0', 'MYR100', '21.67136');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-08', 'M0', 'NZD1', '0.63756');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-08', 'M0', 'SGD1', '0.67467');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-08', 'M0', 'THB100', '2.76201');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-08', 'M0', 'XDR1', '1.29987');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-08', 'M0', 'USD3M', '0.912');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-08', 'M0', 'USD6M', '0.9096');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-09', 'M0', 'EUR1', '1.08595');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-09', 'M0', 'GBP1', '1.26734');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-09', 'M0', 'DKK100', '14.60398');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-09', 'M0', 'NOK100', '10.65417');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-09', 'M0', 'CZK100', '4.27845');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-09', 'M0', 'HUF100', '0.30827');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-09', 'M0', 'PLN100', '23.77532');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-09', 'M0', 'RUB1', '0.01265');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-09', 'M0', 'SEK100', '10.67276');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-09', 'M0', 'TRY100', '10.7955');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-09', 'M0', 'USD1', '0.92233');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-09', 'M0', 'CAD1', '0.72861');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-09', 'M0', 'ARS1', '0.00939');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-09', 'M0', 'BRL100', '17.52316');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-09', 'M0', 'MXN100', '4.60592');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-09', 'M0', 'ZAR1', '0.06338');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-09', 'M0', 'JPY100', '0.83728');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-09', 'M0', 'AUD1', '0.67549');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-09', 'M0', 'CNY100', '14.28441');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-09', 'M0', 'HKD100', '11.85431');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-09', 'M0', 'KRW100', '0.07867');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-09', 'M0', 'MYR100', '22.13273');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-09', 'M0', 'NZD1', '0.65154');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-09', 'M0', 'SGD1', '0.68449');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-09', 'M0', 'THB100', '2.79031');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-09', 'M0', 'XDR1', '1.31085');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-09', 'M0', 'USD3M', '0.9202');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-09', 'M0', 'USD6M', '0.9178');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-10', 'M0', 'EUR1', '1.07137');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-10', 'M0', 'GBP1', '1.26398');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-10', 'M0', 'DKK100', '14.40046');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-10', 'M0', 'NOK100', '10.90088');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-10', 'M0', 'CZK100', '4.2023');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-10', 'M0', 'HUF100', '0.29682');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-10', 'M0', 'PLN100', '23.32327');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-10', 'M0', 'RUB1', '0.01294');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-10', 'M0', 'SEK100', '10.64842');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-10', 'M0', 'TRY100', '10.04283');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-10', 'M0', 'USD1', '0.92356');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-10', 'M0', 'CAD1', '0.74204');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-10', 'M0', 'ARS1', '0.00931');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-10', 'M0', 'BRL100', '16.70066');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-10', 'M0', 'MXN100', '4.51115');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-10', 'M0', 'ZAR1', '0.06218');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-10', 'M0', 'JPY100', '0.81663');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-10', 'M0', 'AUD1', '0.68326');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-10', 'M0', 'CNY100', '14.38428');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-10', 'M0', 'HKD100', '11.87079');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-10', 'M0', 'KRW100', '0.0781');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-10', 'M0', 'MYR100', '22.18013');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-10', 'M0', 'NZD1', '0.65092');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-10', 'M0', 'SGD1', '0.6836');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-10', 'M0', 'THB100', '2.76053');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-10', 'M0', 'XDR1', '1.30324');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-10', 'M0', 'USD3M', '0.921');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-10', 'M0', 'USD6M', '0.9188');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-11', 'M0', 'EUR1', '1.05217');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-11', 'M0', 'GBP1', '1.24082');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-11', 'M0', 'DKK100', '14.14731');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-11', 'M0', 'NOK100', '10.56299');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-11', 'M0', 'CZK100', '4.14278');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-11', 'M0', 'HUF100', '0.28869');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-11', 'M0', 'PLN100', '22.63431');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-11', 'M0', 'RUB1', '0.01266');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-11', 'M0', 'SEK100', '10.47546');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-11', 'M0', 'TRY100', '8.70018');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-11', 'M0', 'USD1', '0.92184');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-11', 'M0', 'CAD1', '0.73375');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-11', 'M0', 'ARS1', '0.00919');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-11', 'M0', 'BRL100', '16.58544');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-11', 'M0', 'MXN100', '4.41642');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-11', 'M0', 'ZAR1', '0.05941');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-11', 'M0', 'JPY100', '0.80879');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-11', 'M0', 'AUD1', '0.67368');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-11', 'M0', 'CNY100', '14.4293');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-11', 'M0', 'HKD100', '11.8332');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-11', 'M0', 'KRW100', '0.0779');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-11', 'M0', 'MYR100', '22.06395');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-11', 'M0', 'NZD1', '0.64708');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-11', 'M0', 'SGD1', '0.67931');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-11', 'M0', 'THB100', '2.78574');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-11', 'M0', 'XDR1', '1.29376');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-11', 'M0', 'USD3M', '0.9191');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-11', 'M0', 'USD6M', '0.9167');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-12', 'M0', 'EUR1', '1.04078');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-12', 'M0', 'GBP1', '1.2259');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-12', 'M0', 'DKK100', '13.99625');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-12', 'M0', 'NOK100', '10.26689');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-12', 'M0', 'CZK100', '4.12174');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-12', 'M0', 'HUF100', '0.2833');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-12', 'M0', 'PLN100', '22.5478');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-12', 'M0', 'RUB1', '0.01248');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-12', 'M0', 'SEK100', '10.13557');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-12', 'M0', 'TRY100', '6.86317');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-12', 'M0', 'USD1', '0.9209');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-12', 'M0', 'CAD1', '0.71976');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-12', 'M0', 'ARS1', '0.00904');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-12', 'M0', 'BRL100', '16.28731');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-12', 'M0', 'MXN100', '4.40168');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-12', 'M0', 'ZAR1', '0.05802');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-12', 'M0', 'JPY100', '0.80808');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-12', 'M0', 'AUD1', '0.65938');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-12', 'M0', 'CNY100', '14.46188');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-12', 'M0', 'HKD100', '11.80821');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-12', 'M0', 'KRW100', '0.07778');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-12', 'M0', 'MYR100', '21.87046');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-12', 'M0', 'NZD1', '0.62512');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-12', 'M0', 'SGD1', '0.67564');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-12', 'M0', 'THB100', '2.74434');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-12', 'M0', 'XDR1', '1.28692');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-12', 'M0', 'USD3M', '0.9183');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2021-12', 'M0', 'USD6M', '0.9157');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-01', 'M0', 'EUR1', '1.04018');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-01', 'M0', 'GBP1', '1.24588');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-01', 'M0', 'DKK100', '13.97931');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-01', 'M0', 'NOK100', '10.39293');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-01', 'M0', 'CZK100', '4.2518');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-01', 'M0', 'HUF100', '0.29001');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-01', 'M0', 'PLN100', '22.84826');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-01', 'M0', 'RUB1', '0.01202');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-01', 'M0', 'SEK100', '10.05149');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-01', 'M0', 'TRY100', '6.77205');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-01', 'M0', 'USD1', '0.91868');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-01', 'M0', 'CAD1', '0.72822');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-01', 'M0', 'ARS1', '0.00885');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-01', 'M0', 'BRL100', '16.58074');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-01', 'M0', 'MXN100', '4.48265');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-01', 'M0', 'ZAR1', '0.05935');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-01', 'M0', 'JPY100', '0.79952');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-01', 'M0', 'AUD1', '0.65982');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-01', 'M0', 'CNY100', '14.45474');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-01', 'M0', 'HKD100', '11.78955');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-01', 'M0', 'KRW100', '0.07682');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-01', 'M0', 'MYR100', '21.9261');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-01', 'M0', 'NZD1', '0.61981');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-01', 'M0', 'SGD1', '0.68021');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-01', 'M0', 'THB100', '2.76562');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-01', 'M0', 'XDR1', '1.28656');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-01', 'M0', 'USD3M', '0.9162');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-01', 'M0', 'USD6M', '0.9132');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-02', 'M0', 'EUR1', '1.04611');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-02', 'M0', 'GBP1', '1.24925');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-02', 'M0', 'DKK100', '14.05885');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-02', 'M0', 'NOK100', '10.39822');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-02', 'M0', 'CZK100', '4.28312');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-02', 'M0', 'HUF100', '0.29317');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-02', 'M0', 'PLN100', '22.99345');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-02', 'M0', 'RUB1', '0.01183');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-02', 'M0', 'SEK100', '9.92367');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-02', 'M0', 'TRY100', '6.7559');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-02', 'M0', 'USD1', '0.92266');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-02', 'M0', 'CAD1', '0.7253');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-02', 'M0', 'ARS1', '0.00869');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-02', 'M0', 'BRL100', '17.75218');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-02', 'M0', 'MXN100', '4.50944');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-02', 'M0', 'ZAR1', '0.0606');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-02', 'M0', 'JPY100', '0.80117');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-02', 'M0', 'AUD1', '0.66085');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-02', 'M0', 'CNY100', '14.54277');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-02', 'M0', 'HKD100', '11.82933');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-02', 'M0', 'KRW100', '0.07697');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-02', 'M0', 'MYR100', '22.03235');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-02', 'M0', 'NZD1', '0.6157');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-02', 'M0', 'SGD1', '0.68504');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-02', 'M0', 'THB100', '2.82734');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-02', 'M0', 'XDR1', '1.29276');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-02', 'M0', 'USD3M', '0.9197');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-02', 'M0', 'USD6M', '0.9157');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-03', 'M0', 'EUR1', '1.02458');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-03', 'M0', 'GBP1', '1.22444');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-03', 'M0', 'DKK100', '13.77083');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-03', 'M0', 'NOK100', '10.52065');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-03', 'M0', 'CZK100', '4.10213');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-03', 'M0', 'HUF100', '0.27214');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-03', 'M0', 'PLN100', '21.55273');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-03', 'M0', 'RUB1', '0.00872');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-03', 'M0', 'SEK100', '9.7168');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-03', 'M0', 'TRY100', '6.35487');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-03', 'M0', 'USD1', '0.92967');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-03', 'M0', 'CAD1', '0.73428');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-03', 'M0', 'ARS1', '0.00851');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-03', 'M0', 'BRL100', '18.65366');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-03', 'M0', 'MXN100', '4.51944');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-03', 'M0', 'ZAR1', '0.06202');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-03', 'M0', 'JPY100', '0.78397');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-03', 'M0', 'AUD1', '0.68516');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-03', 'M0', 'CNY100', '14.65311');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-03', 'M0', 'HKD100', '11.88409');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-03', 'M0', 'KRW100', '0.07615');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-03', 'M0', 'MYR100', '22.13094');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-03', 'M0', 'NZD1', '0.63815');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-03', 'M0', 'SGD1', '0.68398');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-03', 'M0', 'THB100', '2.79513');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-03', 'M0', 'XDR1', '1.2849');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-03', 'M0', 'USD3M', '0.926');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-03', 'M0', 'USD6M', '0.9209');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-04', 'M0', 'EUR1', '1.02172');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-04', 'M0', 'GBP1', '1.22122');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-04', 'M0', 'DKK100', '13.73474');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-04', 'M0', 'NOK100', '10.61003');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-04', 'M0', 'CZK100', '4.18104');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-04', 'M0', 'HUF100', '0.27238');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-04', 'M0', 'PLN100', '21.97446');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-04', 'M0', 'RUB1', '0.01191');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-04', 'M0', 'SEK100', '9.90104');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-04', 'M0', 'TRY100', '6.41082');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-04', 'M0', 'USD1', '0.94391');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-04', 'M0', 'CAD1', '0.74814');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-04', 'M0', 'ARS1', '0.00834');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-04', 'M0', 'BRL100', '19.92795');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-04', 'M0', 'MXN100', '4.7026');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-04', 'M0', 'ZAR1', '0.06282');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-04', 'M0', 'JPY100', '0.7473');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-04', 'M0', 'AUD1', '0.69682');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-04', 'M0', 'CNY100', '14.66993');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-04', 'M0', 'HKD100', '12.03787');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-04', 'M0', 'KRW100', '0.07641');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-04', 'M0', 'MYR100', '22.09732');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-04', 'M0', 'NZD1', '0.63928');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-04', 'M0', 'SGD1', '0.69102');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-04', 'M0', 'THB100', '2.79307');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-04', 'M0', 'XDR1', '1.2891');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-04', 'M0', 'USD3M', '0.9394');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-04', 'M0', 'USD6M', '0.9331');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-05', 'M0', 'EUR1', '1.03591');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-05', 'M0', 'GBP1', '1.21942');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-05', 'M0', 'DKK100', '13.92306');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-05', 'M0', 'NOK100', '10.21936');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-05', 'M0', 'CZK100', '4.18452');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-05', 'M0', 'HUF100', '0.26978');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-05', 'M0', 'PLN100', '22.26913');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-05', 'M0', 'RUB1', '0.01523');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-05', 'M0', 'SEK100', '9.87572');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-05', 'M0', 'TRY100', '6.2913');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-05', 'M0', 'USD1', '0.98024');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-05', 'M0', 'CAD1', '0.76282');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-05', 'M0', 'ARS1', '0.00834');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-05', 'M0', 'BRL100', '19.73513');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-05', 'M0', 'MXN100', '4.88533');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-05', 'M0', 'ZAR1', '0.06161');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-05', 'M0', 'JPY100', '0.76041');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-05', 'M0', 'AUD1', '0.69113');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-05', 'M0', 'CNY100', '14.63506');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-05', 'M0', 'HKD100', '12.48849');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-05', 'M0', 'KRW100', '0.07732');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-05', 'M0', 'MYR100', '22.37555');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-05', 'M0', 'NZD1', '0.6274');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-05', 'M0', 'SGD1', '0.70881');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-05', 'M0', 'THB100', '2.84735');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-05', 'M0', 'XDR1', '1.31461');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-05', 'M0', 'USD3M', '0.974');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-05', 'M0', 'USD6M', '0.967');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-06', 'M0', 'EUR1', '1.02495');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-06', 'M0', 'GBP1', '1.19485');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-06', 'M0', 'DKK100', '13.77732');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-06', 'M0', 'NOK100', '9.94607');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-06', 'M0', 'CZK100', '4.14539');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-06', 'M0', 'HUF100', '0.25822');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-06', 'M0', 'PLN100', '22.05405');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-06', 'M0', 'RUB1', '0.01703');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-06', 'M0', 'SEK100', '9.66685');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-06', 'M0', 'TRY100', '5.69668');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-06', 'M0', 'USD1', '0.96962');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-06', 'M0', 'CAD1', '0.75647');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-06', 'M0', 'ARS1', '0.00791');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-06', 'M0', 'BRL100', '19.2346');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-06', 'M0', 'MXN100', '4.84534');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-06', 'M0', 'ZAR1', '0.06138');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-06', 'M0', 'JPY100', '0.72381');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-06', 'M0', 'AUD1', '0.68082');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-06', 'M0', 'CNY100', '14.47895');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-06', 'M0', 'HKD100', '12.35462');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-06', 'M0', 'KRW100', '0.07582');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-06', 'M0', 'MYR100', '22.02477');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-06', 'M0', 'NZD1', '0.61565');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-06', 'M0', 'SGD1', '0.70067');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-06', 'M0', 'THB100', '2.77386');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-06', 'M0', 'XDR1', '1.2963');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-06', 'M0', 'USD3M', '0.9631');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-06', 'M0', 'USD6M', '0.9558');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-07', 'M0', 'EUR1', '0.98765');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-07', 'M0', 'GBP1', '1.16254');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-07', 'M0', 'DKK100', '13.2707');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-07', 'M0', 'NOK100', '9.68793');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-07', 'M0', 'CZK100', '4.01505');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-07', 'M0', 'HUF100', '0.24444');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-07', 'M0', 'PLN100', '20.69005');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-07', 'M0', 'RUB1', '0.01642');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-07', 'M0', 'SEK100', '9.33381');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-07', 'M0', 'TRY100', '5.54941');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-07', 'M0', 'USD1', '0.96954');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-07', 'M0', 'CAD1', '0.74919');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-07', 'M0', 'ARS1', '0.00757');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-07', 'M0', 'BRL100', '18.04921');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-07', 'M0', 'MXN100', '4.72285');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-07', 'M0', 'ZAR1', '0.05752');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-07', 'M0', 'JPY100', '0.70908');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-07', 'M0', 'AUD1', '0.66437');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-07', 'M0', 'CNY100', '14.40007');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-07', 'M0', 'HKD100', '12.35213');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-07', 'M0', 'KRW100', '0.07414');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-07', 'M0', 'MYR100', '21.83229');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-07', 'M0', 'NZD1', '0.60063');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-07', 'M0', 'SGD1', '0.69464');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-07', 'M0', 'THB100', '2.66648');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-07', 'M0', 'XDR1', '1.27828');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-07', 'M0', 'USD3M', '0.9626');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-07', 'M0', 'USD6M', '0.954');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-08', 'M0', 'EUR1', '0.96899');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-08', 'M0', 'GBP1', '1.14693');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-08', 'M0', 'DKK100', '13.0253');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-08', 'M0', 'NOK100', '9.8602');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-08', 'M0', 'CZK100', '3.94514');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-08', 'M0', 'HUF100', '0.24092');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-08', 'M0', 'PLN100', '20.52874');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-08', 'M0', 'RUB1', '0.01576');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-08', 'M0', 'SEK100', '9.21974');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-08', 'M0', 'TRY100', '5.30527');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-08', 'M0', 'USD1', '0.95717');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-08', 'M0', 'CAD1', '0.74116');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-08', 'M0', 'ARS1', '0.00709');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-08', 'M0', 'BRL100', '18.61382');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-08', 'M0', 'MXN100', '4.75566');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-08', 'M0', 'ZAR1', '0.05734');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-08', 'M0', 'JPY100', '0.70772');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-08', 'M0', 'AUD1', '0.66613');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-08', 'M0', 'CNY100', '14.06791');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-08', 'M0', 'HKD100', '12.1992');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-08', 'M0', 'KRW100', '0.07249');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-08', 'M0', 'MYR100', '21.42617');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-08', 'M0', 'NZD1', '0.59962');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-08', 'M0', 'SGD1', '0.69142');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-08', 'M0', 'THB100', '2.67159');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-08', 'M0', 'XDR1', '1.2583');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-08', 'M0', 'USD3M', '0.9497');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-08', 'M0', 'USD6M', '0.9411');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-09', 'M0', 'EUR1', '0.96419');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-09', 'M0', 'GBP1', '1.10347');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-09', 'M0', 'DKK100', '12.96564');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-09', 'M0', 'NOK100', '9.50279');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-09', 'M0', 'CZK100', '3.92288');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-09', 'M0', 'HUF100', '0.23873');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-09', 'M0', 'PLN100', '20.32739');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-09', 'M0', 'RUB1', '0.0162');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-09', 'M0', 'SEK100', '8.94482');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-09', 'M0', 'TRY100', '5.31038');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-09', 'M0', 'USD1', '0.97293');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-09', 'M0', 'CAD1', '0.73088');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-09', 'M0', 'ARS1', '0.0068');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-09', 'M0', 'BRL100', '18.62051');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-09', 'M0', 'MXN100', '4.84742');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-09', 'M0', 'ZAR1', '0.05544');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-09', 'M0', 'JPY100', '0.68075');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-09', 'M0', 'AUD1', '0.65041');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-09', 'M0', 'CNY100', '13.85476');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-09', 'M0', 'HKD100', '12.39519');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-09', 'M0', 'KRW100', '0.06972');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-09', 'M0', 'MYR100', '21.40682');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-09', 'M0', 'NZD1', '0.57777');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-09', 'M0', 'SGD1', '0.68832');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-09', 'M0', 'THB100', '2.62679');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-09', 'M0', 'XDR1', '1.25636');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-09', 'M0', 'USD3M', '0.9651');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-09', 'M0', 'USD6M', '0.9566');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-10', 'M0', 'EUR1', '0.97899');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-10', 'M0', 'GBP1', '1.12398');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-10', 'M0', 'DKK100', '13.16062');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-10', 'M0', 'NOK100', '9.41809');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-10', 'M0', 'CZK100', '3.98994');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-10', 'M0', 'HUF100', '0.23419');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-10', 'M0', 'PLN100', '20.36972');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-10', 'M0', 'RUB1', '0.01614');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-10', 'M0', 'SEK100', '8.94175');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-10', 'M0', 'TRY100', '5.35473');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-10', 'M0', 'USD1', '0.99564');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-10', 'M0', 'CAD1', '0.72599');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-10', 'M0', 'ARS1', '0.00656');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-10', 'M0', 'BRL100', '18.91138');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-10', 'M0', 'MXN100', '4.9786');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-10', 'M0', 'ZAR1', '0.05492');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-10', 'M0', 'JPY100', '0.67652');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-10', 'M0', 'AUD1', '0.63286');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-10', 'M0', 'CNY100', '13.83964');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-10', 'M0', 'HKD100', '12.68395');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-10', 'M0', 'KRW100', '0.06979');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-10', 'M0', 'MYR100', '21.2069');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-10', 'M0', 'NZD1', '0.56614');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-10', 'M0', 'SGD1', '0.6987');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-10', 'M0', 'THB100', '2.6245');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-10', 'M0', 'XDR1', '1.27656');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-10', 'M0', 'USD3M', '0.9848');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-10', 'M0', 'USD6M', '0.9751');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-11', 'M0', 'EUR1', '0.9843');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-11', 'M0', 'GBP1', '1.13286');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-11', 'M0', 'DKK100', '13.23148');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-11', 'M0', 'NOK100', '9.51697');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-11', 'M0', 'CZK100', '4.03946');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-11', 'M0', 'HUF100', '0.24179');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-11', 'M0', 'PLN100', '20.96683');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-11', 'M0', 'RUB1', '0.01584');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-11', 'M0', 'SEK100', '9.04706');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-11', 'M0', 'TRY100', '5.18798');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-11', 'M0', 'USD1', '0.96588');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-11', 'M0', 'CAD1', '0.71855');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-11', 'M0', 'ARS1', '0.00598');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-11', 'M0', 'BRL100', '18.32133');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-11', 'M0', 'MXN100', '4.96159');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-11', 'M0', 'ZAR1', '0.05519');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-11', 'M0', 'JPY100', '0.67855');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-11', 'M0', 'AUD1', '0.63698');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-11', 'M0', 'CNY100', '13.45553');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-11', 'M0', 'HKD100', '12.33599');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-11', 'M0', 'KRW100', '0.07112');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-11', 'M0', 'MYR100', '20.96009');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-11', 'M0', 'NZD1', '0.58513');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-11', 'M0', 'SGD1', '0.69628');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-11', 'M0', 'THB100', '2.65458');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-11', 'M0', 'XDR1', '1.2543');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-11', 'M0', 'USD3M', '0.9549');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-11', 'M0', 'USD6M', '0.9448');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-12', 'M0', 'EUR1', '0.98654');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-12', 'M0', 'GBP1', '1.13487');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-12', 'M0', 'DKK100', '13.26406');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-12', 'M0', 'NOK100', '9.43884');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-12', 'M0', 'CZK100', '4.06462');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-12', 'M0', 'HUF100', '0.24231');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-12', 'M0', 'PLN100', '21.06493');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-12', 'M0', 'RUB1', '0.01415');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-12', 'M0', 'SEK100', '8.98384');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-12', 'M0', 'TRY100', '4.99415');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-12', 'M0', 'USD1', '0.93205');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-12', 'M0', 'CAD1', '0.68622');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-12', 'M0', 'ARS1', '0.00542');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-12', 'M0', 'BRL100', '17.76749');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-12', 'M0', 'MXN100', '4.75481');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-12', 'M0', 'ZAR1', '0.05398');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-12', 'M0', 'JPY100', '0.69079');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-12', 'M0', 'AUD1', '0.62919');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-12', 'M0', 'CNY100', '13.35874');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-12', 'M0', 'HKD100', '11.97113');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-12', 'M0', 'KRW100', '0.07205');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-12', 'M0', 'MYR100', '21.12426');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-12', 'M0', 'NZD1', '0.59208');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-12', 'M0', 'SGD1', '0.68937');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-12', 'M0', 'THB100', '2.67903');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-12', 'M0', 'XDR1', '1.238');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-12', 'M0', 'USD3M', '0.9223');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2022-12', 'M0', 'USD6M', '0.9129');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-01', 'M0', 'EUR1', '0.99615');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-01', 'M0', 'GBP1', '1.1296');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-01', 'M0', 'DKK100', '13.39202');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-01', 'M0', 'NOK100', '9.3013');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-01', 'M0', 'CZK100', '4.15871');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-01', 'M0', 'HUF100', '0.2519');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-01', 'M0', 'PLN100', '21.19802');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-01', 'M0', 'RUB1', '0.01328');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-01', 'M0', 'SEK100', '8.89424');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-01', 'M0', 'TRY100', '4.91825');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-01', 'M0', 'USD1', '0.92411');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-01', 'M0', 'CAD1', '0.68812');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-01', 'M0', 'ARS1', '0.00508');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-01', 'M0', 'BRL100', '17.79058');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-01', 'M0', 'MXN100', '4.87077');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-01', 'M0', 'ZAR1', '0.05403');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-01', 'M0', 'JPY100', '0.70877');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-01', 'M0', 'AUD1', '0.64224');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-01', 'M0', 'CNY100', '13.60888');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-01', 'M0', 'HKD100', '11.81537');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-01', 'M0', 'KRW100', '0.07437');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-01', 'M0', 'MYR100', '21.3763');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-01', 'M0', 'NZD1', '0.5913');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-01', 'M0', 'SGD1', '0.69731');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-01', 'M0', 'THB100', '2.78207');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-01', 'M0', 'XDR1', '1.24262');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-01', 'M0', 'USD3M', '0.9152');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-01', 'M0', 'USD6M', '0.9064');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-02', 'M0', 'EUR1', '0.99045');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-02', 'M0', 'GBP1', '1.11787');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-02', 'M0', 'DKK100', '13.30322');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-02', 'M0', 'NOK100', '9.04817');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-02', 'M0', 'CZK100', '4.17426');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-02', 'M0', 'HUF100', '0.25737');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-02', 'M0', 'PLN100', '20.88237');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-02', 'M0', 'RUB1', '0.01261');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-02', 'M0', 'SEK100', '8.86279');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-02', 'M0', 'TRY100', '4.90303');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-02', 'M0', 'USD1', '0.92427');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-02', 'M0', 'CAD1', '0.68785');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-02', 'M0', 'ARS1', '0.00483');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-02', 'M0', 'BRL100', '17.89159');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-02', 'M0', 'MXN100', '4.96248');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-02', 'M0', 'ZAR1', '0.05164');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-02', 'M0', 'JPY100', '0.6958');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-02', 'M0', 'AUD1', '0.63852');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-02', 'M0', 'CNY100', '13.52017');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-02', 'M0', 'HKD100', '11.77914');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-02', 'M0', 'KRW100', '0.07243');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-02', 'M0', 'MYR100', '21.16146');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-02', 'M0', 'NZD1', '0.58216');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-02', 'M0', 'SGD1', '0.69441');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-02', 'M0', 'THB100', '2.71518');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-02', 'M0', 'XDR1', '1.23844');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-02', 'M0', 'USD3M', '0.9155');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-02', 'M0', 'USD6M', '0.9066');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-03', 'M0', 'EUR1', '0.99051');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-03', 'M0', 'GBP1', '1.12331');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-03', 'M0', 'DKK100', '13.30277');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-03', 'M0', 'NOK100', '8.78479');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-03', 'M0', 'CZK100', '4.18025');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-03', 'M0', 'HUF100', '0.25728');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-03', 'M0', 'PLN100', '21.11561');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-03', 'M0', 'RUB1', '0.01213');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-03', 'M0', 'SEK100', '8.82636');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-03', 'M0', 'TRY100', '4.86973');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-03', 'M0', 'USD1', '0.92552');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-03', 'M0', 'CAD1', '0.67641');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-03', 'M0', 'ARS1', '0.00457');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-03', 'M0', 'BRL100', '17.76368');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-03', 'M0', 'MXN100', '5.02742');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-03', 'M0', 'ZAR1', '0.0506');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-03', 'M0', 'JPY100', '0.69225');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-03', 'M0', 'AUD1', '0.61791');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-03', 'M0', 'CNY100', '13.41733');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-03', 'M0', 'HKD100', '11.79177');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-03', 'M0', 'KRW100', '0.07089');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-03', 'M0', 'MYR100', '20.72094');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-03', 'M0', 'NZD1', '0.57444');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-03', 'M0', 'SGD1', '0.6902');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-03', 'M0', 'THB100', '2.68352');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-03', 'M0', 'XDR1', '1.23632');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-03', 'M0', 'USD3M', '0.9159');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-03', 'M0', 'USD6M', '0.9076');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-04', 'M0', 'EUR1', '0.98479');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-04', 'M0', 'GBP1', '1.11735');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-04', 'M0', 'DKK100', '13.21569');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-04', 'M0', 'NOK100', '8.54636');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-04', 'M0', 'CZK100', '4.20058');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-04', 'M0', 'HUF100', '0.26199');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-04', 'M0', 'PLN100', '21.23988');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-04', 'M0', 'RUB1', '0.01107');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-04', 'M0', 'SEK100', '8.68226');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-04', 'M0', 'TRY100', '4.63803');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-04', 'M0', 'USD1', '0.89768');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-04', 'M0', 'CAD1', '0.66574');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-04', 'M0', 'ARS1', '0.00416');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-04', 'M0', 'BRL100', '17.89625');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-04', 'M0', 'MXN100', '4.96304');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-04', 'M0', 'ZAR1', '0.0494');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-04', 'M0', 'JPY100', '0.672');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-04', 'M0', 'AUD1', '0.60022');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-04', 'M0', 'CNY100', '13.0293');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-04', 'M0', 'HKD100', '11.43582');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-04', 'M0', 'KRW100', '0.06788');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-04', 'M0', 'MYR100', '20.27909');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-04', 'M0', 'NZD1', '0.55697');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-04', 'M0', 'SGD1', '0.67397');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-04', 'M0', 'THB100', '2.62203');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-04', 'M0', 'XDR1', '1.21172');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-04', 'M0', 'USD3M', '0.8894');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-04', 'M0', 'USD6M', '0.8815');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-05', 'M0', 'EUR1', '0.97533');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-05', 'M0', 'GBP1', '1.1199');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-05', 'M0', 'DKK100', '13.09446');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-05', 'M0', 'NOK100', '8.3128');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-05', 'M0', 'CZK100', '4.13438');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-05', 'M0', 'HUF100', '0.26186');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-05', 'M0', 'PLN100', '21.50253');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-05', 'M0', 'RUB1', '0.01136');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-05', 'M0', 'SEK100', '8.58591');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-05', 'M0', 'TRY100', '4.54068');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-05', 'M0', 'USD1', '0.89699');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-05', 'M0', 'CAD1', '0.66373');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-05', 'M0', 'ARS1', '0.00389');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-05', 'M0', 'BRL100', '18.02273');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-05', 'M0', 'MXN100', '5.05497');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-05', 'M0', 'ZAR1', '0.0472');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-05', 'M0', 'JPY100', '0.6553');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-05', 'M0', 'AUD1', '0.59669');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-05', 'M0', 'CNY100', '12.84086');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-05', 'M0', 'HKD100', '11.44547');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-05', 'M0', 'KRW100', '0.06761');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-05', 'M0', 'MYR100', '19.86371');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-05', 'M0', 'NZD1', '0.55788');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-05', 'M0', 'SGD1', '0.66999');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-05', 'M0', 'THB100', '2.62406');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-05', 'M0', 'XDR1', '1.20408');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-05', 'M0', 'USD3M', '0.8879');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-05', 'M0', 'USD6M', '0.8797');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-06', 'M0', 'EUR1', '0.97602');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-06', 'M0', 'GBP1', '1.13719');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-06', 'M0', 'DKK100', '13.10236');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-06', 'M0', 'NOK100', '8.33852');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-06', 'M0', 'CZK100', '4.11796');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-06', 'M0', 'HUF100', '0.26329');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-06', 'M0', 'PLN100', '21.87033');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-06', 'M0', 'RUB1', '0.01078');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-06', 'M0', 'SEK100', '8.36268');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-06', 'M0', 'TRY100', '3.81098');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-06', 'M0', 'USD1', '0.90083');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-06', 'M0', 'CAD1', '0.67764');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-06', 'M0', 'ARS1', '0.00363');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-06', 'M0', 'BRL100', '18.53437');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-06', 'M0', 'MXN100', '5.21931');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-06', 'M0', 'ZAR1', '0.0481');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-06', 'M0', 'JPY100', '0.63779');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-06', 'M0', 'AUD1', '0.60452');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-06', 'M0', 'CNY100', '12.57606');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-06', 'M0', 'HKD100', '11.50173');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-06', 'M0', 'KRW100', '0.06948');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-06', 'M0', 'MYR100', '19.44907');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-06', 'M0', 'NZD1', '0.55226');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-06', 'M0', 'SGD1', '0.66902');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-06', 'M0', 'THB100', '2.57994');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-06', 'M0', 'XDR1', '1.20113');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-06', 'M0', 'USD3M', '0.892');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-06', 'M0', 'USD6M', '0.8835');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-07', 'M0', 'EUR1', '0.9661');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-07', 'M0', 'GBP1', '1.12533');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-07', 'M0', 'DKK100', '12.96625');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-07', 'M0', 'NOK100', '8.50804');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-07', 'M0', 'CZK100', '4.04406');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-07', 'M0', 'HUF100', '0.25497');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-07', 'M0', 'PLN100', '21.74485');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-07', 'M0', 'RUB1', '0.00966');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-07', 'M0', 'SEK100', '8.30282');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-07', 'M0', 'TRY100', '3.29707');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-07', 'M0', 'USD1', '0.87398');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-07', 'M0', 'CAD1', '0.66112');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-07', 'M0', 'ARS1', '0.00329');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-07', 'M0', 'BRL100', '18.18351');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-07', 'M0', 'MXN100', '5.16188');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-07', 'M0', 'ZAR1', '0.04803');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-07', 'M0', 'JPY100', '0.61949');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-07', 'M0', 'AUD1', '0.58847');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-07', 'M0', 'CNY100', '12.15522');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-07', 'M0', 'HKD100', '11.18089');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-07', 'M0', 'KRW100', '0.0681');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-07', 'M0', 'MYR100', '19.03768');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-07', 'M0', 'NZD1', '0.54412');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-07', 'M0', 'SGD1', '0.65485');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-07', 'M0', 'THB100', '2.52524');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-07', 'M0', 'XDR1', '1.17381');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-07', 'M0', 'USD3M', '0.8654');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-07', 'M0', 'USD6M', '0.8564');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-08', 'M0', 'EUR1', '0.95835');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-08', 'M0', 'GBP1', '1.11603');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-08', 'M0', 'DKK100', '12.86015');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-08', 'M0', 'NOK100', '8.38882');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-08', 'M0', 'CZK100', '3.97315');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-08', 'M0', 'HUF100', '0.24872');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-08', 'M0', 'PLN100', '21.47783');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-08', 'M0', 'RUB1', '0.00919');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-08', 'M0', 'SEK100', '8.1092');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-08', 'M0', 'TRY100', '3.25731');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-08', 'M0', 'USD1', '0.87859');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-08', 'M0', 'CAD1', '0.65156');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-08', 'M0', 'ARS1', '0.00276');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-08', 'M0', 'BRL100', '17.91147');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-08', 'M0', 'MXN100', '5.16954');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-08', 'M0', 'ZAR1', '0.04676');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-08', 'M0', 'JPY100', '0.60668');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-08', 'M0', 'AUD1', '0.56924');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-08', 'M0', 'CNY100', '12.10677');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-08', 'M0', 'HKD100', '11.22558');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-08', 'M0', 'KRW100', '0.06638');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-08', 'M0', 'MYR100', '19.03935');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-08', 'M0', 'NZD1', '0.52617');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-08', 'M0', 'SGD1', '0.65007');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-08', 'M0', 'THB100', '2.50709');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-08', 'M0', 'XDR1', '1.17098');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-08', 'M0', 'USD3M', '0.8698');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-08', 'M0', 'USD6M', '0.8606');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-09', 'M0', 'EUR1', '0.95969');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-09', 'M0', 'GBP1', '1.11365');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-09', 'M0', 'DKK100', '12.87107');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-09', 'M0', 'NOK100', '8.3766');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-09', 'M0', 'CZK100', '3.93741');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-09', 'M0', 'HUF100', '0.24833');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-09', 'M0', 'PLN100', '20.87711');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-09', 'M0', 'RUB1', '0.00929');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-09', 'M0', 'SEK100', '8.10249');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-09', 'M0', 'TRY100', '3.32477');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-09', 'M0', 'USD1', '0.89823');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-09', 'M0', 'CAD1', '0.66347');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-09', 'M0', 'ARS1', '0.00257');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-09', 'M0', 'BRL100', '18.17607');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-09', 'M0', 'MXN100', '5.1918');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-09', 'M0', 'ZAR1', '0.04734');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-09', 'M0', 'JPY100', '0.60817');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-09', 'M0', 'AUD1', '0.57701');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-09', 'M0', 'CNY100', '12.30869');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-09', 'M0', 'HKD100', '11.47406');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-09', 'M0', 'KRW100', '0.06735');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-09', 'M0', 'MYR100', '19.18531');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-09', 'M0', 'NZD1', '0.5326');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-09', 'M0', 'SGD1', '0.65892');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-09', 'M0', 'THB100', '2.50546');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-09', 'M0', 'XDR1', '1.18616');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-09', 'M0', 'USD3M', '0.8895');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-09', 'M0', 'USD6M', '0.8803');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-10', 'M0', 'EUR1', '0.95518');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-10', 'M0', 'GBP1', '1.10018');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-10', 'M0', 'DKK100', '12.80336');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-10', 'M0', 'NOK100', '8.22272');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-10', 'M0', 'CZK100', '3.88537');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-10', 'M0', 'HUF100', '0.24786');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-10', 'M0', 'PLN100', '21.16628');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-10', 'M0', 'RUB1', '0.00934');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-10', 'M0', 'SEK100', '8.20288');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-10', 'M0', 'TRY100', '3.24237');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-10', 'M0', 'USD1', '0.90385');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-10', 'M0', 'CAD1', '0.65986');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-10', 'M0', 'ARS1', '0.00258');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-10', 'M0', 'BRL100', '17.8534');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-10', 'M0', 'MXN100', '5.00761');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-10', 'M0', 'ZAR1', '0.04743');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-10', 'M0', 'JPY100', '0.60428');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-10', 'M0', 'AUD1', '0.57427');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-10', 'M0', 'CNY100', '12.36889');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-10', 'M0', 'HKD100', '11.55126');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-10', 'M0', 'KRW100', '0.06687');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-10', 'M0', 'MYR100', '19.03957');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-10', 'M0', 'NZD1', '0.53332');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-10', 'M0', 'SGD1', '0.66025');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-10', 'M0', 'THB100', '2.47618');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-10', 'M0', 'XDR1', '1.1857');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-10', 'M0', 'USD3M', '0.8945');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-10', 'M0', 'USD6M', '0.8855');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-11', 'M0', 'EUR1', '0.96324');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-11', 'M0', 'GBP1', '1.10675');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-11', 'M0', 'DKK100', '12.91541');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-11', 'M0', 'NOK100', '8.16249');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-11', 'M0', 'CZK100', '3.93438');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-11', 'M0', 'HUF100', '0.25408');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-11', 'M0', 'PLN100', '21.88145');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-11', 'M0', 'RUB1', '0.00987');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-11', 'M0', 'SEK100', '8.33915');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-11', 'M0', 'TRY100', '3.11173');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-11', 'M0', 'USD1', '0.8918');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-11', 'M0', 'CAD1', '0.64981');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-11', 'M0', 'ARS1', '0.00253');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-11', 'M0', 'BRL100', '18.17612');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-11', 'M0', 'MXN100', '5.12372');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-11', 'M0', 'ZAR1', '0.04808');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-11', 'M0', 'JPY100', '0.59514');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-11', 'M0', 'AUD1', '0.57891');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-11', 'M0', 'CNY100', '12.33963');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-11', 'M0', 'HKD100', '11.42279');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-11', 'M0', 'KRW100', '0.06819');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-11', 'M0', 'MYR100', '19.01988');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-11', 'M0', 'NZD1', '0.53409');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-11', 'M0', 'SGD1', '0.66104');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-11', 'M0', 'THB100', '2.51409');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-11', 'M0', 'XDR1', '1.17952');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-11', 'M0', 'USD3M', '0.8828');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-11', 'M0', 'USD6M', '0.8742');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-12', 'M0', 'EUR1', '0.94415');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-12', 'M0', 'GBP1', '1.09558');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-12', 'M0', 'DKK100', '12.66354');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-12', 'M0', 'NOK100', '8.1897');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-12', 'M0', 'CZK100', '3.85846');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-12', 'M0', 'HUF100', '0.24733');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-12', 'M0', 'PLN100', '21.79434');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-12', 'M0', 'RUB1', '0.00952');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-12', 'M0', 'SEK100', '8.42523');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-12', 'M0', 'TRY100', '2.97656');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-12', 'M0', 'USD1', '0.86603');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-12', 'M0', 'CAD1', '0.64442');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-12', 'M0', 'ARS1', '0.0017');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-12', 'M0', 'BRL100', '17.66104');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-12', 'M0', 'MXN100', '5.03243');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-12', 'M0', 'ZAR1', '0.04646');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-12', 'M0', 'JPY100', '0.60067');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-12', 'M0', 'AUD1', '0.57869');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-12', 'M0', 'CNY100', '12.12405');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-12', 'M0', 'HKD100', '11.08977');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-12', 'M0', 'KRW100', '0.06636');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-12', 'M0', 'MYR100', '18.57613');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-12', 'M0', 'NZD1', '0.53796');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-12', 'M0', 'SGD1', '0.64977');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-12', 'M0', 'THB100', '2.47366');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-12', 'M0', 'XDR1', '1.1544');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-12', 'M0', 'USD3M', '0.8567');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2023-12', 'M0', 'USD6M', '0.8485');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-01', 'M0', 'EUR1', '0.93645');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-01', 'M0', 'GBP1', '1.0911');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-01', 'M0', 'DKK100', '12.55771');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-01', 'M0', 'NOK100', '8.2532');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-01', 'M0', 'CZK100', '3.7888');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-01', 'M0', 'HUF100', '0.24508');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-01', 'M0', 'PLN100', '21.44783');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-01', 'M0', 'RUB1', '0.00963');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-01', 'M0', 'SEK100', '8.29486');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-01', 'M0', 'TRY100', '2.85205');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-01', 'M0', 'USD1', '0.85893');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-01', 'M0', 'CAD1', '0.64004');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-01', 'M0', 'ARS1', '0.00105');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-01', 'M0', 'BRL100', '17.47957');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-01', 'M0', 'MXN100', '5.02927');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-01', 'M0', 'ZAR1', '0.04566');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-01', 'M0', 'JPY100', '0.58702');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-01', 'M0', 'AUD1', '0.57019');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-01', 'M0', 'CNY100', '11.97519');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-01', 'M0', 'HKD100', '10.98783');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-01', 'M0', 'KRW100', '0.06473');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-01', 'M0', 'MYR100', '18.30878');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-01', 'M0', 'NZD1', '0.53002');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-01', 'M0', 'SGD1', '0.64281');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-01', 'M0', 'THB100', '2.43661');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-01', 'M0', 'XDR1', '1.14366');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-01', 'M0', 'USD3M', '0.8494');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-01', 'M0', 'USD6M', '0.8417');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-02', 'M0', 'EUR1', '0.946');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-02', 'M0', 'GBP1', '1.10711');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-02', 'M0', 'DKK100', '12.68899');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-02', 'M0', 'NOK100', '8.30688');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-02', 'M0', 'CZK100', '3.74971');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-02', 'M0', 'HUF100', '0.24375');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-02', 'M0', 'PLN100', '21.86062');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-02', 'M0', 'RUB1', '0.00955');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-02', 'M0', 'SEK100', '8.40986');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-02', 'M0', 'TRY100', '2.84515');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-02', 'M0', 'USD1', '0.87667');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-02', 'M0', 'CAD1', '0.64976');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-02', 'M0', 'ARS1', '0.00105');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-02', 'M0', 'BRL100', '17.66882');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-02', 'M0', 'MXN100', '5.12924');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-02', 'M0', 'ZAR1', '0.04611');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-02', 'M0', 'JPY100', '0.58649');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-02', 'M0', 'AUD1', '0.57223');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-02', 'M0', 'CNY100', '12.18672');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-02', 'M0', 'HKD100', '11.20811');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-02', 'M0', 'KRW100', '0.06584');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-02', 'M0', 'MYR100', '18.38726');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-02', 'M0', 'NZD1', '0.53695');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-02', 'M0', 'SGD1', '0.65197');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-02', 'M0', 'THB100', '2.44528');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-02', 'M0', 'XDR1', '1.16334');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-02', 'M0', 'USD3M', '0.8681');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-02', 'M0', 'USD6M', '0.8598');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-03', 'M0', 'EUR1', '0.96527');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-03', 'M0', 'GBP1', '1.12857');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-03', 'M0', 'DKK100', '12.94548');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-03', 'M0', 'NOK100', '8.38177');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-03', 'M0', 'CZK100', '3.81642');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-03', 'M0', 'HUF100', '0.24429');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-03', 'M0', 'PLN100', '22.40791');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-03', 'M0', 'RUB1', '0.00968');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-03', 'M0', 'SEK100', '8.54012');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-03', 'M0', 'TRY100', '2.77122');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-03', 'M0', 'USD1', '0.8879');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-03', 'M0', 'CAD1', '0.65566');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-03', 'M0', 'ARS1', '0.00105');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-03', 'M0', 'BRL100', '17.8384');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-03', 'M0', 'MXN100', '5.28545');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-03', 'M0', 'ZAR1', '0.04708');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-03', 'M0', 'JPY100', '0.59308');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-03', 'M0', 'AUD1', '0.58202');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-03', 'M0', 'CNY100', '12.32845');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-03', 'M0', 'HKD100', '11.3506');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-03', 'M0', 'KRW100', '0.06671');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-03', 'M0', 'MYR100', '18.8261');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-03', 'M0', 'NZD1', '0.54028');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-03', 'M0', 'SGD1', '0.6624');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-03', 'M0', 'THB100', '2.47106');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-03', 'M0', 'XDR1', '1.18159');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-03', 'M0', 'USD3M', '0.8796');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-03', 'M0', 'USD6M', '0.8709');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-04', 'M0', 'EUR1', '0.97598');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-04', 'M0', 'GBP1', '1.13956');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-04', 'M0', 'DKK100', '13.08354');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-04', 'M0', 'NOK100', '8.35451');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-04', 'M0', 'CZK100', '3.86082');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-04', 'M0', 'HUF100', '0.24865');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-04', 'M0', 'PLN100', '22.68381');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-04', 'M0', 'RUB1', '0.00977');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-04', 'M0', 'SEK100', '8.4224');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-04', 'M0', 'TRY100', '2.81272');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-04', 'M0', 'USD1', '0.90975');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-04', 'M0', 'CAD1', '0.66565');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-04', 'M0', 'ARS1', '0.00105');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-04', 'M0', 'BRL100', '17.75941');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-04', 'M0', 'MXN100', '5.42064');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-04', 'M0', 'ZAR1', '0.04821');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-04', 'M0', 'JPY100', '0.59156');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-04', 'M0', 'AUD1', '0.59254');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-04', 'M0', 'CNY100', '12.56652');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-04', 'M0', 'HKD100', '11.61756');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-04', 'M0', 'KRW100', '0.06645');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-04', 'M0', 'MYR100', '19.0796');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-04', 'M0', 'NZD1', '0.5424');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-04', 'M0', 'SGD1', '0.67066');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-04', 'M0', 'THB100', '2.47412');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-04', 'M0', 'XDR1', '1.20062');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-04', 'M0', 'USD3M', '0.9');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-04', 'M0', 'USD6M', '0.8906');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-05', 'M0', 'EUR1', '0.9828');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-05', 'M0', 'GBP1', '1.14898');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-05', 'M0', 'DKK100', '13.17365');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-05', 'M0', 'NOK100', '8.47458');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-05', 'M0', 'CZK100', '3.9596');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-05', 'M0', 'HUF100', '0.25376');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-05', 'M0', 'PLN100', '22.94823');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-05', 'M0', 'RUB1', '0.01002');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-05', 'M0', 'SEK100', '8.45885');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-05', 'M0', 'TRY100', '2.81851');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-05', 'M0', 'USD1', '0.90912');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-05', 'M0', 'CAD1', '0.665');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-05', 'M0', 'ARS1', '0.00103');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-05', 'M0', 'BRL100', '17.6874');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-05', 'M0', 'MXN100', '5.41322');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-05', 'M0', 'ZAR1', '0.04935');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-05', 'M0', 'JPY100', '0.58319');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-05', 'M0', 'AUD1', '0.60223');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-05', 'M0', 'CNY100', '12.56685');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-05', 'M0', 'HKD100', '11.63808');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-05', 'M0', 'KRW100', '0.06661');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-05', 'M0', 'MYR100', '19.27862');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-05', 'M0', 'NZD1', '0.5517');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-05', 'M0', 'SGD1', '0.67294');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-05', 'M0', 'THB100', '2.48133');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-05', 'M0', 'XDR1', '1.20179');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-05', 'M0', 'USD3M', '0.9001');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-05', 'M0', 'USD6M', '0.8908');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-06', 'M0', 'EUR1', '0.9619');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-06', 'M0', 'GBP1', '1.13646');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-06', 'M0', 'DKK100', '12.89532');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-06', 'M0', 'NOK100', '8.42565');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-06', 'M0', 'CZK100', '3.88326');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-06', 'M0', 'HUF100', '0.24384');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-06', 'M0', 'PLN100', '22.26512');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-06', 'M0', 'RUB1', '0.01019');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-06', 'M0', 'SEK100', '8.52495');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-06', 'M0', 'TRY100', '2.74282');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-06', 'M0', 'USD1', '0.89391');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-06', 'M0', 'CAD1', '0.65205');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-06', 'M0', 'ARS1', '0.00099');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-06', 'M0', 'BRL100', '16.62053');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-06', 'M0', 'MXN100', '4.91336');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-06', 'M0', 'ZAR1', '0.04857');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-06', 'M0', 'JPY100', '0.56619');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-06', 'M0', 'AUD1', '0.59373');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-06', 'M0', 'CNY100', '12.32098');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-06', 'M0', 'HKD100', '11.44657');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-06', 'M0', 'KRW100', '0.06481');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-06', 'M0', 'MYR100', '18.97707');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-06', 'M0', 'NZD1', '0.54843');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-06', 'M0', 'SGD1', '0.66122');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-06', 'M0', 'THB100', '2.43506');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-06', 'M0', 'XDR1', '1.179');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-06', 'M0', 'USD3M', '0.8843');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-06', 'M0', 'USD6M', '0.8751');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-07', 'M0', 'EUR1', '0.96782');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-07', 'M0', 'GBP1', '1.14718');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-07', 'M0', 'DKK100', '12.97271');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-07', 'M0', 'NOK100', '8.27054');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-07', 'M0', 'CZK100', '3.82691');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-07', 'M0', 'HUF100', '0.24645');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-07', 'M0', 'PLN100', '22.60184');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-07', 'M0', 'RUB1', '0.01021');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-07', 'M0', 'SEK100', '8.39229');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-07', 'M0', 'TRY100', '2.71117');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-07', 'M0', 'USD1', '0.89244');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-07', 'M0', 'CAD1', '0.65102');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-07', 'M0', 'ARS1', '0.00097');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-07', 'M0', 'BRL100', '16.10825');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-07', 'M0', 'MXN100', '4.92697');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-07', 'M0', 'ZAR1', '0.04894');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-07', 'M0', 'JPY100', '0.56548');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-07', 'M0', 'AUD1', '0.59562');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-07', 'M0', 'CNY100', '12.28777');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-07', 'M0', 'HKD100', '11.4281');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-07', 'M0', 'KRW100', '0.06452');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-07', 'M0', 'MYR100', '19.07512');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-07', 'M0', 'NZD1', '0.53837');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-07', 'M0', 'SGD1', '0.66276');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-07', 'M0', 'THB100', '2.46096');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-07', 'M0', 'XDR1', '1.18021');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-07', 'M0', 'USD3M', '0.8827');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-07', 'M0', 'USD6M', '0.8732');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-08', 'M0', 'EUR1', '0.94494');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-08', 'M0', 'GBP1', '1.10943');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-08', 'M0', 'DKK100', '12.66426');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-08', 'M0', 'NOK100', '8.01461');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-08', 'M0', 'CZK100', '3.75497');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-08', 'M0', 'HUF100', '0.23933');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-08', 'M0', 'PLN100', '22.01625');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-08', 'M0', 'RUB1', '0.00958');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-08', 'M0', 'SEK100', '8.25208');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-08', 'M0', 'TRY100', '2.54181');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-08', 'M0', 'USD1', '0.85732');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-08', 'M0', 'CAD1', '0.62763');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-08', 'M0', 'ARS1', '0.00091');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-08', 'M0', 'BRL100', '15.44905');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-08', 'M0', 'MXN100', '4.47658');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-08', 'M0', 'ZAR1', '0.04754');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-08', 'M0', 'JPY100', '0.58674');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-08', 'M0', 'AUD1', '0.57081');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-08', 'M0', 'CNY100', '11.99358');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-08', 'M0', 'HKD100', '10.99883');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-08', 'M0', 'KRW100', '0.06345');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-08', 'M0', 'MYR100', '19.45343');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-08', 'M0', 'NZD1', '0.5217');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-08', 'M0', 'SGD1', '0.65198');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-08', 'M0', 'THB100', '2.46884');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-08', 'M0', 'XDR1', '1.15054');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-08', 'M0', 'USD3M', '0.8488');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-08', 'M0', 'USD6M', '0.8404');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-09', 'M0', 'EUR1', '0.94101');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-09', 'M0', 'GBP1', '1.11956');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-09', 'M0', 'DKK100', '12.61399');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-09', 'M0', 'NOK100', '7.98581');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-09', 'M0', 'CZK100', '3.74907');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-09', 'M0', 'HUF100', '0.23828');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-09', 'M0', 'PLN100', '21.99999');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-09', 'M0', 'RUB1', '0.00926');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-09', 'M0', 'SEK100', '8.28394');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-09', 'M0', 'TRY100', '2.48702');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-09', 'M0', 'USD1', '0.84719');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-09', 'M0', 'CAD1', '0.6258');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-09', 'M0', 'ARS1', '0.00088');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-09', 'M0', 'BRL100', '15.27289');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-09', 'M0', 'MXN100', '4.32247');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-09', 'M0', 'ZAR1', '0.0481');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-09', 'M0', 'JPY100', '0.59182');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-09', 'M0', 'AUD1', '0.57361');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-09', 'M0', 'CNY100', '11.96862');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-09', 'M0', 'HKD100', '10.87318');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-09', 'M0', 'KRW100', '0.06362');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-09', 'M0', 'MYR100', '19.87728');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-09', 'M0', 'NZD1', '0.52716');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-09', 'M0', 'SGD1', '0.65357');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-09', 'M0', 'THB100', '2.54235');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-09', 'M0', 'XDR1', '1.14415');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-09', 'M0', 'USD3M', '0.8384');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-09', 'M0', 'USD6M', '0.8309');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-10', 'M0', 'EUR1', '0.93867');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-10', 'M0', 'GBP1', '1.12372');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-10', 'M0', 'DKK100', '12.58383');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-10', 'M0', 'NOK100', '7.95869');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-10', 'M0', 'CZK100', '3.71029');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-10', 'M0', 'HUF100', '0.23367');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-10', 'M0', 'PLN100', '21.74647');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-10', 'M0', 'RUB1', '0.00892');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-10', 'M0', 'SEK100', '8.23001');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-10', 'M0', 'TRY100', '2.51244');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-10', 'M0', 'USD1', '0.86059');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-10', 'M0', 'CAD1', '0.62595');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-10', 'M0', 'ARS1', '0.00088');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-10', 'M0', 'BRL100', '15.34413');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-10', 'M0', 'MXN100', '4.37229');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-10', 'M0', 'ZAR1', '0.049');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-10', 'M0', 'JPY100', '0.57536');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-10', 'M0', 'AUD1', '0.5775');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-10', 'M0', 'CNY100', '12.14234');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-10', 'M0', 'HKD100', '11.07543');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-10', 'M0', 'KRW100', '0.06323');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-10', 'M0', 'MYR100', '20.01728');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-10', 'M0', 'NZD1', '0.52375');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-10', 'M0', 'SGD1', '0.65711');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-10', 'M0', 'THB100', '2.57707');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-10', 'M0', 'XDR1', '1.15022');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-10', 'M0', 'USD3M', '0.8517');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-10', 'M0', 'USD6M', '0.8437');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-11', 'M0', 'EUR1', '0.9354');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-11', 'M0', 'GBP1', '1.12169');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-11', 'M0', 'DKK100', '12.54173');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-11', 'M0', 'NOK100', '7.96342');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-11', 'M0', 'CZK100', '3.6963');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-11', 'M0', 'HUF100', '0.22843');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-11', 'M0', 'PLN100', '21.57432');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-11', 'M0', 'RUB1', '0.00872');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-11', 'M0', 'SEK100', '8.07525');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-11', 'M0', 'TRY100', '2.5544');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-11', 'M0', 'USD1', '0.88034');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-11', 'M0', 'CAD1', '0.62994');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-11', 'M0', 'ARS1', '0.00088');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-11', 'M0', 'BRL100', '15.1956');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-11', 'M0', 'MXN100', '4.32976');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-11', 'M0', 'ZAR1', '0.04911');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-11', 'M0', 'JPY100', '0.57306');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-11', 'M0', 'AUD1', '0.57504');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-11', 'M0', 'CNY100', '12.21271');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-11', 'M0', 'HKD100', '11.31575');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-11', 'M0', 'KRW100', '0.06311');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-11', 'M0', 'MYR100', '19.84594');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-11', 'M0', 'NZD1', '0.52062');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-11', 'M0', 'SGD1', '0.65852');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-11', 'M0', 'THB100', '2.55552');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-11', 'M0', 'XDR1', '1.1607');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-11', 'M0', 'USD3M', '0.8711');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-11', 'M0', 'USD6M', '0.8624');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-12', 'M0', 'EUR1', '0.93383');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-12', 'M0', 'GBP1', '1.12782');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-12', 'M0', 'DKK100', '12.52015');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-12', 'M0', 'NOK100', '7.95615');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-12', 'M0', 'CZK100', '3.71591');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-12', 'M0', 'HUF100', '0.22677');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-12', 'M0', 'PLN100', '21.86461');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-12', 'M0', 'RUB1', '0.00861');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-12', 'M0', 'SEK100', '8.1154');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-12', 'M0', 'TRY100', '2.54723');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-12', 'M0', 'USD1', '0.89113');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-12', 'M0', 'CAD1', '0.62614');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-12', 'M0', 'ARS1', '0.00088');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-12', 'M0', 'BRL100', '14.6259');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-12', 'M0', 'MXN100', '4.39728');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-12', 'M0', 'ZAR1', '0.04904');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-12', 'M0', 'JPY100', '0.57998');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-12', 'M0', 'AUD1', '0.56522');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-12', 'M0', 'CNY100', '12.23695');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-12', 'M0', 'HKD100', '11.46391');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-12', 'M0', 'KRW100', '0.06197');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-12', 'M0', 'MYR100', '19.98138');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-12', 'M0', 'NZD1', '0.51296');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-12', 'M0', 'SGD1', '0.66009');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-12', 'M0', 'THB100', '2.60849');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-12', 'M0', 'XDR1', '1.16838');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-12', 'M0', 'USD3M', '0.8827');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2024-12', 'M0', 'USD6M', '0.8734');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-01', 'M0', 'EUR1', '0.94162');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-01', 'M0', 'GBP1', '1.12201');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-01', 'M0', 'DKK100', '12.62055');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-01', 'M0', 'NOK100', '8.01414');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-01', 'M0', 'CZK100', '3.74265');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-01', 'M0', 'HUF100', '0.22862');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-01', 'M0', 'PLN100', '22.17575');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-01', 'M0', 'RUB1', '0.00895');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-01', 'M0', 'SEK100', '8.2002');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-01', 'M0', 'TRY100', '2.55688');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-01', 'M0', 'USD1', '0.90933');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-01', 'M0', 'CAD1', '0.63189');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-01', 'M0', 'ARS1', '0.00087');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-01', 'M0', 'BRL100', '15.1065');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-01', 'M0', 'MXN100', '4.42126');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-01', 'M0', 'ZAR1', '0.04856');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-01', 'M0', 'JPY100', '0.58142');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-01', 'M0', 'AUD1', '0.56641');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-01', 'M0', 'CNY100', '12.45953');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-01', 'M0', 'HKD100', '11.67982');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-01', 'M0', 'KRW100', '0.06263');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-01', 'M0', 'MYR100', '20.36653');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-01', 'M0', 'NZD1', '0.51208');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-01', 'M0', 'SGD1', '0.66813');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-01', 'M0', 'THB100', '2.65448');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-01', 'M0', 'XDR1', '1.18247');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-01', 'M0', 'USD3M', '0.8996');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-01', 'M0', 'USD6M', '0.8902');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-02', 'M0', 'EUR1', '0.94114');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-02', 'M0', 'GBP1', '1.13291');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-02', 'M0', 'DKK100', '12.61708');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-02', 'M0', 'NOK100', '8.0731');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-02', 'M0', 'CZK100', '3.75307');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-02', 'M0', 'HUF100', '0.23344');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-02', 'M0', 'PLN100', '22.55001');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-02', 'M0', 'RUB1', '0.00982');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-02', 'M0', 'SEK100', '8.36523');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-02', 'M0', 'TRY100', '2.49694');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-02', 'M0', 'USD1', '0.90396');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-02', 'M0', 'CAD1', '0.632');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-02', 'M0', 'ARS1', '0.00085');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-02', 'M0', 'BRL100', '15.68208');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-02', 'M0', 'MXN100', '4.4123');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-02', 'M0', 'ZAR1', '0.04889');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-02', 'M0', 'JPY100', '0.59539');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-02', 'M0', 'AUD1', '0.56951');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-02', 'M0', 'CNY100', '12.42677');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-02', 'M0', 'HKD100', '11.61589');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-02', 'M0', 'KRW100', '0.06255');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-02', 'M0', 'MYR100', '20.34668');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-02', 'M0', 'NZD1', '0.51338');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-02', 'M0', 'SGD1', '0.6712');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-02', 'M0', 'THB100', '2.67538');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-02', 'M0', 'XDR1', '1.18209');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-02', 'M0', 'USD3M', '0.8945');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-02', 'M0', 'USD6M', '0.885');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-03', 'M0', 'EUR1', '0.95488');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-03', 'M0', 'GBP1', '1.14062');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-03', 'M0', 'DKK100', '12.80071');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-03', 'M0', 'NOK100', '8.26861');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-03', 'M0', 'CZK100', '3.81862');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-03', 'M0', 'HUF100', '0.23889');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-03', 'M0', 'PLN100', '22.8318');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-03', 'M0', 'RUB1', '0.01029');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-03', 'M0', 'SEK100', '8.708');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-03', 'M0', 'TRY100', '2.37529');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-03', 'M0', 'USD1', '0.88351');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-03', 'M0', 'CAD1', '0.61557');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-03', 'M0', 'ARS1', '0.00083');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-03', 'M0', 'BRL100', '15.32263');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-03', 'M0', 'MXN100', '4.36776');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-03', 'M0', 'ZAR1', '0.04836');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-03', 'M0', 'JPY100', '0.59254');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-03', 'M0', 'AUD1', '0.5566');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-03', 'M0', 'CNY100', '12.18519');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-03', 'M0', 'HKD100', '11.36625');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-03', 'M0', 'KRW100', '0.06061');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-03', 'M0', 'MYR100', '19.92189');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-03', 'M0', 'NZD1', '0.50581');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-03', 'M0', 'SGD1', '0.66125');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-03', 'M0', 'THB100', '2.61351');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-03', 'M0', 'XDR1', '1.17323');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-03', 'M0', 'USD3M', '0.874');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-03', 'M0', 'USD6M', '0.865');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-04', 'M0', 'EUR1', '0.93799');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-04', 'M0', 'GBP1', '1.09923');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-04', 'M0', 'DKK100', '12.56531');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-04', 'M0', 'NOK100', '7.93362');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-04', 'M0', 'CZK100', '3.745');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-04', 'M0', 'HUF100', '0.23071');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-04', 'M0', 'PLN100', '21.99663');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-04', 'M0', 'RUB1', '0.01003');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-04', 'M0', 'SEK100', '8.55182');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-04', 'M0', 'TRY100', '2.19389');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-04', 'M0', 'USD1', '0.83684');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-04', 'M0', 'CAD1', '0.59769');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-04', 'M0', 'ARS1', '0.00075');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-04', 'M0', 'BRL100', '14.4949');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-04', 'M0', 'MXN100', '4.16871');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-04', 'M0', 'ZAR1', '0.04428');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-04', 'M0', 'JPY100', '0.57933');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-04', 'M0', 'AUD1', '0.52599');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-04', 'M0', 'CNY100', '11.46118');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-04', 'M0', 'HKD100', '10.77888');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-04', 'M0', 'KRW100', '0.05797');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-04', 'M0', 'MYR100', '18.9478');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-04', 'M0', 'NZD1', '0.48727');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-04', 'M0', 'SGD1', '0.63166');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-04', 'M0', 'THB100', '2.47505');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-04', 'M0', 'XDR1', '1.12462');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-04', 'M0', 'USD3M', '0.8254');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-04', 'M0', 'USD6M', '0.8167');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-05', 'M0', 'EUR1', '0.93578');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-05', 'M0', 'GBP1', '1.10814');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-05', 'M0', 'DKK100', '12.54397');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-05', 'M0', 'NOK100', '8.06451');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-05', 'M0', 'CZK100', '3.75546');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-05', 'M0', 'HUF100', '0.23168');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-05', 'M0', 'PLN100', '21.99494');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-05', 'M0', 'RUB1', '0.01028');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-05', 'M0', 'SEK100', '8.59568');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-05', 'M0', 'TRY100', '2.13684');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-05', 'M0', 'USD1', '0.82921');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-05', 'M0', 'CAD1', '0.59794');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-05', 'M0', 'ARS1', '0.00072');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-05', 'M0', 'BRL100', '14.63379');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-05', 'M0', 'MXN100', '4.26665');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-05', 'M0', 'ZAR1', '0.0458');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-05', 'M0', 'JPY100', '0.57335');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-05', 'M0', 'AUD1', '0.53417');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-05', 'M0', 'CNY100', '11.49497');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-05', 'M0', 'HKD100', '10.62893');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-05', 'M0', 'KRW100', '0.05963');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-05', 'M0', 'MYR100', '19.44773');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-05', 'M0', 'NZD1', '0.49242');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-05', 'M0', 'SGD1', '0.6409');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-05', 'M0', 'THB100', '2.5179');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-05', 'M0', 'XDR1', '1.12173');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-05', 'M0', 'USD3M', '0.8199');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-05', 'M0', 'USD6M', '0.8106');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-06', 'M0', 'EUR1', '0.93766');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-06', 'M0', 'GBP1', '1.10318');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-06', 'M0', 'DKK100', '12.57');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-06', 'M0', 'NOK100', '8.09352');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-06', 'M0', 'CZK100', '3.78079');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-06', 'M0', 'HUF100', '0.23323');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-06', 'M0', 'PLN100', '21.98641');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-06', 'M0', 'RUB1', '0.01033');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-06', 'M0', 'SEK100', '8.5199');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-06', 'M0', 'TRY100', '2.062');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-06', 'M0', 'USD1', '0.81376');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-06', 'M0', 'CAD1', '0.59489');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-06', 'M0', 'ARS1', '0.00069');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-06', 'M0', 'BRL100', '14.65737');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-06', 'M0', 'MXN100', '4.27288');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-06', 'M0', 'ZAR1', '0.0456');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-06', 'M0', 'JPY100', '0.5633');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-06', 'M0', 'AUD1', '0.52878');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-06', 'M0', 'CNY100', '11.33098');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-06', 'M0', 'HKD100', '10.36815');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-06', 'M0', 'KRW100', '0.0596');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-06', 'M0', 'MYR100', '19.18247');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-06', 'M0', 'NZD1', '0.49033');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-06', 'M0', 'SGD1', '0.63418');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-06', 'M0', 'THB100', '2.4959');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-06', 'M0', 'XDR1', '1.1109');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-06', 'M0', 'USD3M', '0.8046');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-06', 'M0', 'USD6M', '0.7956');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-07', 'M0', 'EUR1', '0.93245');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-07', 'M0', 'GBP1', '1.07794');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-07', 'M0', 'DKK100', '12.49525');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-07', 'M0', 'NOK100', '7.86063');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-07', 'M0', 'CZK100', '3.78631');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-07', 'M0', 'HUF100', '0.23359');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-07', 'M0', 'PLN100', '21.91656');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-07', 'M0', 'RUB1', '0.01011');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-07', 'M0', 'SEK100', '8.32884');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-07', 'M0', 'TRY100', '1.9824');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-07', 'M0', 'USD1', '0.79786');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-07', 'M0', 'CAD1', '0.58305');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-07', 'M0', 'ARS1', '0.00063');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-07', 'M0', 'BRL100', '14.44395');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-07', 'M0', 'MXN100', '4.26807');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-07', 'M0', 'ZAR1', '0.04492');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-07', 'M0', 'JPY100', '0.54317');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-07', 'M0', 'AUD1', '0.52177');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-07', 'M0', 'CNY100', '11.12389');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-07', 'M0', 'HKD100', '10.16417');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-07', 'M0', 'KRW100', '0.05792');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-07', 'M0', 'MYR100', '18.83314');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-07', 'M0', 'NZD1', '0.47869');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-07', 'M0', 'SGD1', '0.6229');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-07', 'M0', 'THB100', '2.45959');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-07', 'M0', 'XDR1', '1.09367');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-07', 'M0', 'USD3M', '0.7887');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-07', 'M0', 'USD6M', '0.7799');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-08', 'M0', 'EUR1', '0.93875');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-08', 'M0', 'GBP1', '1.08509');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-08', 'M0', 'DKK100', '12.57762');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-08', 'M0', 'NOK100', '7.90953');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-08', 'M0', 'CZK100', '3.82938');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-08', 'M0', 'HUF100', '0.23683');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-08', 'M0', 'PLN100', '22.02424');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-08', 'M0', 'RUB1', '0.01006');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-08', 'M0', 'SEK100', '8.41276');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-08', 'M0', 'TRY100', '1.97389');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-08', 'M0', 'USD1', '0.80649');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-08', 'M0', 'CAD1', '0.58429');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-08', 'M0', 'ARS1', '0.00061');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-08', 'M0', 'BRL100', '14.80997');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-08', 'M0', 'MXN100', '4.31003');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-08', 'M0', 'ZAR1', '0.04555');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-08', 'M0', 'JPY100', '0.54669');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-08', 'M0', 'AUD1', '0.52363');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-08', 'M0', 'CNY100', '11.24384');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-08', 'M0', 'HKD100', '10.30556');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-08', 'M0', 'KRW100', '0.05804');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-08', 'M0', 'MYR100', '19.09008');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-08', 'M0', 'NZD1', '0.47573');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-08', 'M0', 'SGD1', '0.6276');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-08', 'M0', 'THB100', '2.48664');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-08', 'M0', 'XDR1', '1.10101');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-08', 'M0', 'USD3M', '0.7979');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-08', 'M0', 'USD6M', '0.7894');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-09', 'M0', 'EUR1', '0.93483');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-09', 'M0', 'GBP1', '1.07581');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-09', 'M0', 'DKK100', '12.52379');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-09', 'M0', 'NOK100', '8.01101');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-09', 'M0', 'CZK100', '3.83955');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-09', 'M0', 'HUF100', '0.23862');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-09', 'M0', 'PLN100', '21.94901');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-09', 'M0', 'RUB1', '0.00962');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-09', 'M0', 'SEK100', '8.49996');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-09', 'M0', 'TRY100', '1.92697');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-09', 'M0', 'USD1', '0.79661');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-09', 'M0', 'CAD1', '0.57605');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-09', 'M0', 'ARS1', '0.00057');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-09', 'M0', 'BRL100', '14.82875');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-09', 'M0', 'MXN100', '4.30615');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-09', 'M0', 'ZAR1', '0.04565');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-09', 'M0', 'JPY100', '0.53878');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-09', 'M0', 'AUD1', '0.52527');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-09', 'M0', 'CNY100', '11.18146');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-09', 'M0', 'HKD100', '10.23243');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-09', 'M0', 'KRW100', '0.05721');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-09', 'M0', 'MYR100', '18.90983');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-09', 'M0', 'NZD1', '0.46888');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-09', 'M0', 'SGD1', '0.62009');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-09', 'M0', 'THB100', '2.48948');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-09', 'M0', 'XDR1', '1.09138');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-09', 'M0', 'USD3M', '0.7881');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-09', 'M0', 'USD6M', '0.7804');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-10', 'M0', 'EUR1', '0.92857');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-10', 'M0', 'GBP1', '1.0655');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-10', 'M0', 'DKK100', '12.43428');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-10', 'M0', 'NOK100', '7.95535');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-10', 'M0', 'CZK100', '3.81913');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-10', 'M0', 'HUF100', '0.23808');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-10', 'M0', 'PLN100', '21.85281');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-10', 'M0', 'RUB1', '0.00988');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-10', 'M0', 'SEK100', '8.46117');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-10', 'M0', 'TRY100', '1.90612');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-10', 'M0', 'USD1', '0.79772');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-10', 'M0', 'CAD1', '0.57011');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-10', 'M0', 'ARS1', '0.00056');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-10', 'M0', 'BRL100', '14.81056');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-10', 'M0', 'MXN100', '4.32807');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-10', 'M0', 'ZAR1', '0.04615');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-10', 'M0', 'JPY100', '0.52708');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-10', 'M0', 'AUD1', '0.52197');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-10', 'M0', 'CNY100', '11.20319');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-10', 'M0', 'HKD100', '10.26035');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-10', 'M0', 'KRW100', '0.05605');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-10', 'M0', 'MYR100', '18.91929');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-10', 'M0', 'NZD1', '0.45956');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-10', 'M0', 'SGD1', '0.6157');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-10', 'M0', 'THB100', '2.45007');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-10', 'M0', 'XDR1', '1.08857');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-10', 'M0', 'USD3M', '0.7892');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-10', 'M0', 'USD6M', '0.7819');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-11', 'M0', 'EUR1', '0.92887');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-11', 'M0', 'GBP1', '1.05529');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-11', 'M0', 'DKK100', '12.43825');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-11', 'M0', 'NOK100', '7.91461');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-11', 'M0', 'CZK100', '3.83201');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-11', 'M0', 'HUF100', '0.24166');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-11', 'M0', 'PLN100', '21.91476');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-11', 'M0', 'RUB1', '0.01002');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-11', 'M0', 'SEK100', '8.45338');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-11', 'M0', 'TRY100', '1.9004');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-11', 'M0', 'USD1', '0.80389');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-11', 'M0', 'CAD1', '0.57186');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-11', 'M0', 'ARS1', '0.00056');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-11', 'M0', 'BRL100', '15.04943');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-11', 'M0', 'MXN100', '4.36225');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-11', 'M0', 'ZAR1', '0.04665');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-11', 'M0', 'JPY100', '0.51814');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-11', 'M0', 'AUD1', '0.52285');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-11', 'M0', 'CNY100', '11.30979');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-11', 'M0', 'HKD100', '10.33568');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-11', 'M0', 'KRW100', '0.05504');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-11', 'M0', 'MYR100', '19.35109');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-11', 'M0', 'NZD1', '0.45456');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-11', 'M0', 'SGD1', '0.61691');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-11', 'M0', 'THB100', '2.48152');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-11', 'M0', 'XDR1', '1.09075');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-11', 'M0', 'USD3M', '0.7954');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-11', 'M0', 'USD6M', '0.7878');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-12', 'M0', 'EUR1', '0.93313');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-12', 'M0', 'GBP1', '1.0662');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-12', 'M0', 'DKK100', '12.4928');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-12', 'M0', 'NOK100', '7.88348');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-12', 'M0', 'CZK100', '3.84657');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-12', 'M0', 'HUF100', '0.24249');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-12', 'M0', 'PLN100', '22.08736');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-12', 'M0', 'RUB1', '0.01016');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-12', 'M0', 'SEK100', '8.56667');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-12', 'M0', 'TRY100', '1.86747');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-12', 'M0', 'USD1', '0.79709');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-12', 'M0', 'CAD1', '0.5771');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-12', 'M0', 'ARS1', '0.00055');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-12', 'M0', 'BRL100', '14.63028');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-12', 'M0', 'MXN100', '4.40691');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-12', 'M0', 'ZAR1', '0.0473');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-12', 'M0', 'JPY100', '0.5114');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-12', 'M0', 'AUD1', '0.52921');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-12', 'M0', 'CNY100', '11.31454');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-12', 'M0', 'HKD100', '10.24324');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-12', 'M0', 'KRW100', '0.05433');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-12', 'M0', 'MYR100', '19.47613');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-12', 'M0', 'NZD1', '0.46092');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-12', 'M0', 'SGD1', '0.61719');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-12', 'M0', 'THB100', '2.52151');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-12', 'M0', 'XDR1', '1.08806');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-12', 'M0', 'USD3M', '0.7884');
INSERT INTO currencies("Date", D0, D1, "Value")
VALUES ('2025-12', 'M0', 'USD6M', '0.781');

