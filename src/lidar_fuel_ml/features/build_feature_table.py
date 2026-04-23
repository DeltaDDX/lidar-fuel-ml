"""Feature-table assembly scaffold."""

from __future__ import annotations

from ..schemas import FeatureRecord, PlotPointCloud
from .canopy_metrics import compute_cbh_proxies, compute_plot_summary_metrics
from .vertical_profile import compute_vertical_profile


def build_feature_table(plot_clouds: list[PlotPointCloud]) -> list[FeatureRecord]:
    """Convert extracted plot clouds into model-ready feature rows."""

    feature_rows = []
    for plot_cloud in plot_clouds:
        features = {}
        features.update(compute_plot_summary_metrics(plot_cloud))
        features.update(compute_vertical_profile(plot_cloud))
        features.update(compute_cbh_proxies(plot_cloud))
        feature_rows.append(
            FeatureRecord(
                scene_id=plot_cloud.scene_id,
                record_id=plot_cloud.plot.plot_id,
                features=features,
                target=plot_cloud.plot.target_cbh,
                metadata=plot_cloud.metadata,
            )
        )
    return feature_rows
