from __future__ import annotations

import numpy as np
import pandas as pd

from sentiment_engine.features.text_features import enrich_with_text_features


def test_text_feature_enrichment_accepts_array_backed_urls() -> None:
    events = pd.DataFrame(
        [
            {
                "text_clean": "Tariff deal supports market growth",
                "text_raw": "Tariff deal supports market growth!",
                "urls": np.array(["https://example.com/story"]),
            }
        ]
    )

    enriched = enrich_with_text_features(events)

    assert enriched.iloc[0]["url_domains"] == ["example.com"]
    assert enriched.iloc[0]["rule_sentiment_label"] == "volatility_only"
