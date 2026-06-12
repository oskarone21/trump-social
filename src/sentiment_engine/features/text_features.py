from __future__ import annotations

import pandas as pd

from sentiment_engine.models.rules import classify_text
from sentiment_engine.preprocessing.text import extract_url_domains, punctuation_features


def enrich_with_text_features(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in events.to_dict("records"):
        text_clean = str(row["text_clean"])
        raw_text = str(row["text_raw"])
        classification = classify_text(text_clean)
        features = punctuation_features(raw_text, text_clean)
        urls = row.get("urls") or []
        if isinstance(urls, str):
            urls = [urls]
        row.update(features)
        row["url_domains"] = extract_url_domains(list(urls))
        row["rule_sentiment_label"] = classification.sentiment_label
        row["rule_sentiment_confidence"] = classification.sentiment_confidence
        row["rule_topic_labels"] = classification.topic_labels
        row["rule_topic_confidence"] = classification.topic_confidence
        row["rule_tradeability_label"] = classification.tradeability_label
        row["rule_reason_codes"] = classification.reason_codes
        rows.append(row)
    return pd.DataFrame(rows)
