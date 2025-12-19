from __future__ import annotations
from .data_callbacks import register_data_callbacks
from .earnings_callbacks import register_earnings_callbacks
from .alert_callbacks import register_alert_callbacks
from .explorer_callbacks import register_explorer_callbacks
from .artifact_callbacks import register_artifact_callbacks
from .monitoring_callbacks import register_monitoring_callbacks
from .general_callbacks import register_general_callbacks


def register_all_callbacks(
    app,
    data_dir,
    db_url,
    load_on_start,
    initial_df,
    load_data_csv_first,
    generate_dashboard_artifacts,
):
    register_data_callbacks(
        app, data_dir, db_url, load_on_start, initial_df, load_data_csv_first
    )
    register_earnings_callbacks(app, initial_df)
    register_alert_callbacks(app, initial_df)
    register_explorer_callbacks(app, initial_df)
    register_artifact_callbacks(app, initial_df, generate_dashboard_artifacts)
    register_monitoring_callbacks(app, initial_df, generate_dashboard_artifacts)
    register_general_callbacks(app, initial_df)
