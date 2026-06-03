"""FinBERT sentiment analysis with a graceful offline fallback.

When the ``transformers`` stack and the pre-trained ``ProsusAI/finbert`` model
are available (and the model can be loaded -- it is downloaded on first use),
this module runs proper FinBERT sentiment classification over SEC filing
sections and news headlines.

When that is **not** possible -- no network to download weights, ``torch`` not
installed, or running in a constrained CI/Colab sandbox -- it transparently
falls back to a deterministic finance-tuned lexicon scorer so the rest of the
application (and the test-suite) keeps working with zero degradation in API.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

from freealpharadar.config import settings
from freealpharadar.utils import get_logger

logger = get_logger(__name__)

# A compact finance sentiment lexicon (Loughran-McDonald inspired) used for the
# offline fallback. Not a substitute for FinBERT, but directionally useful.
_POSITIVE = {
    "growth",
    "profit",
    "profitable",
    "record",
    "expansion",
    "strong",
    "beat",
    "exceeded",
    "innovation",
    "leading",
    "breakthrough",
    "award",
    "partnership",
    "surge",
    "robust",
    "upgraded",
    "outperform",
    "milestone",
    "demand",
    "accelerate",
    "momentum",
    "efficient",
    "advantage",
    "win",
    "gain",
}
_NEGATIVE = {
    "loss",
    "decline",
    "lawsuit",
    "litigation",
    "investigation",
    "fraud",
    "weak",
    "miss",
    "missed",
    "downgrade",
    "risk",
    "uncertainty",
    "default",
    "bankruptcy",
    "impairment",
    "restructuring",
    "layoff",
    "recall",
    "breach",
    "delay",
    "shortfall",
    "dilution",
    "going concern",
    "material weakness",
    "adverse",
    "penalty",
    "decrease",
}

_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z'\-]+")


@dataclass
class SentimentResult:
    """A sentiment classification outcome.

    Attributes:
        label: One of ``"positive"``, ``"negative"`` or ``"neutral"``.
        score: Signed sentiment in ``[-1, 1]`` (positive minus negative
            probability for FinBERT; normalised lexicon balance otherwise).
        positive: Probability/weight of the positive class.
        negative: Probability/weight of the negative class.
        neutral: Probability/weight of the neutral class.
        backend: ``"finbert"`` or ``"lexicon"`` -- which engine produced this.
    """

    label: str
    score: float
    positive: float
    negative: float
    neutral: float
    backend: str


class FinBERTSentiment:
    """Lazy-loading FinBERT sentiment analyser with lexicon fallback.

    The heavyweight model is loaded only on first use. If loading fails for any
    reason the analyser permanently switches to the lexicon backend.

    Args:
        model_name: HuggingFace model id. Defaults to the configured FinBERT.
    """

    def __init__(self, model_name: Optional[str] = None) -> None:
        self.model_name = model_name or settings.finbert_model
        self._pipeline = None
        self._tried_load = False
        self._backend = "lexicon"

    # ------------------------------------------------------------------ #
    def _ensure_model(self) -> None:
        """Attempt to load the FinBERT pipeline exactly once."""
        if self._tried_load:
            return
        self._tried_load = True
        if settings.offline:
            logger.info("Offline mode: using lexicon sentiment backend.")
            return
        try:
            from transformers import pipeline  # type: ignore

            self._pipeline = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                tokenizer=self.model_name,
                truncation=True,
                max_length=512,
            )
            self._backend = "finbert"
            logger.info("FinBERT model '%s' loaded.", self.model_name)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "FinBERT unavailable (%s); falling back to lexicon backend.", exc
            )
            self._pipeline = None
            self._backend = "lexicon"

    @property
    def backend(self) -> str:
        """Which backend is in use (after a load attempt)."""
        self._ensure_model()
        return self._backend

    # ------------------------------------------------------------------ #
    def analyze(self, text: str) -> SentimentResult:
        """Classify a single block of text.

        Args:
            text: The text to analyse. Long text is truncated by the tokenizer
                (FinBERT) or scanned wholesale (lexicon).

        Returns:
            A :class:`SentimentResult`.
        """
        if not text or not text.strip():
            return SentimentResult("neutral", 0.0, 0.0, 0.0, 1.0, self._backend)

        self._ensure_model()
        if self._pipeline is not None:
            return self._analyze_finbert(text)
        return self._analyze_lexicon(text)

    def analyze_many(self, texts: List[str]) -> SentimentResult:
        """Analyse several texts and return their averaged sentiment."""
        results = [self.analyze(t) for t in texts if t and t.strip()]
        if not results:
            return SentimentResult("neutral", 0.0, 0.0, 0.0, 1.0, self._backend)
        n = len(results)
        pos = sum(r.positive for r in results) / n
        neg = sum(r.negative for r in results) / n
        neu = sum(r.neutral for r in results) / n
        score = sum(r.score for r in results) / n
        label = _label_from_probs(pos, neg, neu)
        return SentimentResult(label, score, pos, neg, neu, self._backend)

    # ------------------------------------------------------------------ #
    def _analyze_finbert(self, text: str) -> SentimentResult:
        """Run the FinBERT pipeline over (a truncated window of) ``text``."""
        try:
            out = self._pipeline(text[:2000])[0]  # type: ignore[index]
            label = str(out["label"]).lower()
            conf = float(out["score"])
            pos = conf if label == "positive" else 0.0
            neg = conf if label == "negative" else 0.0
            neu = conf if label == "neutral" else 0.0
            score = pos - neg
            return SentimentResult(label, score, pos, neg, neu, "finbert")
        except Exception as exc:  # noqa: BLE001
            logger.debug("FinBERT inference failed (%s); using lexicon.", exc)
            return self._analyze_lexicon(text)

    def _analyze_lexicon(self, text: str) -> SentimentResult:
        """Deterministic lexicon-based sentiment for offline operation."""
        words = [w.lower() for w in _WORD_RE.findall(text)]
        if not words:
            return SentimentResult("neutral", 0.0, 0.0, 0.0, 1.0, "lexicon")
        pos_hits = sum(1 for w in words if w in _POSITIVE)
        neg_hits = sum(1 for w in words if w in _NEGATIVE)
        # Multi-word phrases.
        low = text.lower()
        for phrase in ("going concern", "material weakness"):
            if phrase in low:
                neg_hits += 1
        total = pos_hits + neg_hits
        if total == 0:
            return SentimentResult("neutral", 0.0, 0.0, 0.0, 1.0, "lexicon")
        pos = pos_hits / total
        neg = neg_hits / total
        neu = max(0.0, 1.0 - (total / max(len(words), 1)))
        score = pos - neg
        label = _label_from_probs(pos, neg, neu)
        return SentimentResult(label, score, pos, neg, neu, "lexicon")


def _label_from_probs(pos: float, neg: float, neu: float) -> str:
    """Pick the dominant label from class weights."""
    best = max((pos, "positive"), (neg, "negative"), (neu, "neutral"))
    return best[1]
