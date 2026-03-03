create table analytics.eps_streak_analysis
(
    next_earnings_status       text,
    current_streak             text,
    streak_type                text,
    continuation_probability   text,
    mean_reversion_probability text,
    expected_next_outcome      text,
    prediction_confidence      text,
    dynamic_total_reports      text,
    historical_beat_rate       text,
    gaap_revision_momentum     text
);

alter table analytics.eps_streak_analysis
    owner to postgres;

