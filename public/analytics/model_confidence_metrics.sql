create table analytics.model_confidence_metrics
(
    model_name         text,
    brier_score        double precision,
    log_loss           double precision,
    calibration_error  double precision,
    discrimination_auc double precision,
    overall_confidence double precision
);

alter table analytics.model_confidence_metrics
    owner to postgres;

