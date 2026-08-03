from __future__ import annotations

SENTIMENT_BULLISH = "bullish_market"
SENTIMENT_BEARISH = "bearish_market"
SENTIMENT_GEOPOLITICAL = "geopolitical_risk"
SENTIMENT_NEUTRAL = "neutral"
SENTIMENT_VOLATILITY_ONLY = "volatility_only"
SENTIMENT_LOW_CONFIDENCE = "low_confidence"

TRADEABILITY_DIRECTIONAL = "tradeable_directional"
TRADEABILITY_VOLATILITY_ONLY = "volatility_only"
TRADEABILITY_NO_TRADE_WHIPSAW = "no_trade_whipsaw"
TRADEABILITY_NO_IMPACT = "no_impact"
TRADEABILITY_AMBIGUOUS = "ambiguous"

TOPIC_TRADE_POLICY = "trade_policy"
TOPIC_CHINA_TARIFFS = "china_tariffs"
TOPIC_FED_MONETARY = "fed_monetary"
TOPIC_INFLATION_RATES = "inflation_rates"
TOPIC_IRAN_ENERGY = "iran_energy"
TOPIC_MIDDLE_EAST = "middle_east_geopolitics"
TOPIC_EQUITIES = "equities_direct"
TOPIC_FX_USD = "fx_usd"
TOPIC_GOLD_METALS = "gold_metals"
TOPIC_TECH = "technology_ai_semiconductors"
TOPIC_TAX_FISCAL = "tax_fiscal_policy"
TOPIC_DOMESTIC = "domestic_political"
TOPIC_LEGAL = "legal_personal"
TOPIC_MEDIA = "media_attack"
TOPIC_CAMPAIGN = "campaign_political"
TOPIC_OTHER = "other"

TOPIC_LABELS = [
    TOPIC_TRADE_POLICY,
    TOPIC_CHINA_TARIFFS,
    TOPIC_FED_MONETARY,
    TOPIC_INFLATION_RATES,
    TOPIC_IRAN_ENERGY,
    TOPIC_MIDDLE_EAST,
    TOPIC_EQUITIES,
    TOPIC_FX_USD,
    TOPIC_GOLD_METALS,
    TOPIC_TECH,
    TOPIC_TAX_FISCAL,
    TOPIC_DOMESTIC,
    TOPIC_LEGAL,
    TOPIC_MEDIA,
    TOPIC_CAMPAIGN,
    TOPIC_OTHER,
]

TOPIC_KEYWORDS = {
    TOPIC_TRADE_POLICY: [
        "trade",
        "tariff",
        "tariffs",
        "tarriff",
        "tarriffs",
        "deal",
        "import",
        "export",
    ],
    TOPIC_CHINA_TARIFFS: ["china", "tariff", "tariffs", "beijing", "chinese", "ccp", "xi"],
    TOPIC_FED_MONETARY: ["fed", "powell", "rates", "rate", "cut"],
    TOPIC_INFLATION_RATES: ["inflation", "cpi", "prices"],
    TOPIC_IRAN_ENERGY: ["iran", "oil", "oil prices", "energy", "crude", "opec", "hormuz"],
    TOPIC_MIDDLE_EAST: ["middle east", "israel", "iran", "gaza", "hamas", "hezbollah"],
    TOPIC_EQUITIES: [
        "market",
        "nasdaq",
        "s&p",
        "s&p 500",
        "s&p500",
        "sp500",
        "spx",
        "stocks",
        "equities",
        "dow",
        "russell",
        "vix",
        "futures",
    ],
    TOPIC_FX_USD: ["dollar", "us dollar", "usd", "currency", "currencies", "forex", "fx"],
    TOPIC_GOLD_METALS: ["gold", "silver", "bullion", "precious metals"],
    TOPIC_TECH: ["ai", "semiconductor", "chip", "technology", "tech"],
    TOPIC_TAX_FISCAL: ["tax", "spending", "deficit", "regulation"],
    TOPIC_DOMESTIC: ["crowd", "america", "american"],
    TOPIC_LEGAL: ["court", "judge", "legal"],
    TOPIC_MEDIA: ["media", "fake news"],
    TOPIC_CAMPAIGN: ["campaign", "poll", "election"],
}

BULLISH_KEYWORDS = [
    "boom",
    "beautiful",
    "cut rates",
    "deal",
    "growth",
    "lower regulation",
    "support",
    "fly",
]
BEARISH_KEYWORDS = [
    "tariffs are coming",
    "tariff",
    "tariffs",
    "tarriff",
    "tarriffs",
    "fire",
    "war",
    "sanction",
    "sanctions",
]
GEOPOLITICAL_KEYWORDS = ["iran", "middle east", "israel", "energy security", "fire"]
NEUTRAL_KEYWORDS = ["crowd", "thank you", "fake news", "media"]
