"""
Minimal DAPS Excel segmentation CLI.

Reads a spreadsheet of think-aloud transcripts, applies a lightweight
Dimension-Aware Process Segmentation (DAPS) implementation, and writes an Excel
workbook with segment-level and boundary-level evidence records.
"""

from __future__ import annotations

import argparse
from difflib import SequenceMatcher
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from pandas.api.types import is_object_dtype, is_string_dtype


EVENT_TOKEN = "<EVENT>"
TOKEN_RE = re.compile(r"<EVENT>|[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?|[.!?;:,]")
MOJIBAKE_REPLACEMENT = "\u00ef\u00bf\u00bd"
MOJIBAKE_REPLACEMENT_RE = re.compile(f"(?:{MOJIBAKE_REPLACEMENT})+")
SPEAKER_LABEL_RE = re.compile(
    r"\b(?:Interviewer|Interviewee|Participant|Child|Student|Researcher|Experimenter|"
    r"Subject|Respondent|Speaker)\s*[A-Za-z0-9]*\s*:\s*",
    flags=re.IGNORECASE,
)
CONTROL_TRANSLATION = str.maketrans(
    {
        "\x14": " - ",
        "\x19": "'",
        "\x1c": '"',
        "\x1d": '"',
        "\xfd": "",
        "ý": "",
        "\ufeff": "",
    }
)


@dataclass
class DAPSConfig:
    context_width: int = 12
    local_radius: int = 6
    sensitivity: float = 0.55
    min_segment_tokens: int = 12
    max_segment_tokens: int = 60
    max_segments_per_record: int = 80
    nms_radius: int = 6
    alpha_task_density: float = 0.35
    transition_weight: float = 0.55
    ambiguous_margin: float = 0.04
    cognitive_drop_floor: float = 0.35
    cognitive_drop_ceiling: float = 0.80
    cognitive_semantic_weight: float = 0.45
    cognitive_cue_shift_weight: float = 0.55
    cognitive_semantic_gate_floor: float = 0.25
    legality_mode: str = "hybrid"
    spacy_model: str = "en_core_web_sm"
    semi_legal_penalty: float = 0.18
    sentence_boundary_boost: float = 0.08


def _word_set(text: str | Iterable[str]) -> set[str]:
    if isinstance(text, str):
        parts = re.split(r"[\s,;]+", text)
    else:
        parts = list(text)
    return {str(part).strip().lower() for part in parts if str(part).strip()}


class DefaultVocabulary:
    task = {
        "rotate", "turn", "turned", "flip", "flipped", "fit", "fits", "match",
        "matches", "compare", "align", "put", "make", "same", "equal",
        "together", "apart", "triangle", "triangles", "square", "squares",
        "rectangle", "circle", "shape", "shapes", "piece", "pieces", "line",
        "lines", "angle", "point", "corner", "edge", "top", "bottom", "side",
        "middle", "center", "left", "right", "front", "back", "big", "bigger",
        "small", "smaller", "long", "longer", "short", "wide", "wider",
        "skinny", "skinnier", "straight", "flat", "curve", "size", "sizes",
    }
    cognitive = {
        "rotate", "turn", "turned", "flip", "flipped", "move", "moved", "slide",
        "compare", "match", "matches", "fit", "fits", "align", "put", "try",
        "change", "different", "same", "start", "switch", "instead", "maybe",
        "wait", "actually", "wrong", "nevermind", "draw", "drawing",
    }
    metacognitive = {
        "think", "guess", "maybe", "wait", "wrong", "sure", "check", "know",
        "realize", "realized", "notice", "noticed", "confused", "hmm", "actually",
        "probably", "might", "maybe", "nevermind", "no",
    }
    affective = {
        "frustrating", "frustrated", "ugh", "hard", "easy", "tricky", "confusing",
        "good", "bad", "mad", "laughter", "like", "hate",
    }
    structural = {
        "then", "next", "first", "because", "so", "but", "however", "again",
        "also", "now", "okay",
    }


@dataclass
class SignalVocabulary:
    task: set[str] = field(default_factory=lambda: set(DefaultVocabulary.task))
    cognitive: set[str] = field(default_factory=lambda: set(DefaultVocabulary.cognitive))
    metacognitive: set[str] = field(default_factory=lambda: set(DefaultVocabulary.metacognitive))
    affective: set[str] = field(default_factory=lambda: set(DefaultVocabulary.affective))
    structural: set[str] = field(default_factory=lambda: set(DefaultVocabulary.structural))

    @classmethod
    def from_texts(
        cls,
        task: str | Iterable[str] | None = None,
        cognitive: str | Iterable[str] | None = None,
        metacognitive: str | Iterable[str] | None = None,
        affective: str | Iterable[str] | None = None,
        structural: str | Iterable[str] | None = None,
    ) -> "SignalVocabulary":
        defaults = cls()
        return cls(
            task=_word_set(task) if task is not None else defaults.task,
            cognitive=_word_set(cognitive) if cognitive is not None else defaults.cognitive,
            metacognitive=_word_set(metacognitive) if metacognitive is not None else defaults.metacognitive,
            affective=_word_set(affective) if affective is not None else defaults.affective,
            structural=_word_set(structural) if structural is not None else defaults.structural,
        )

    def as_text(self, name: str) -> str:
        return ", ".join(sorted(getattr(self, name)))


ARTICLES_DETERMINERS = {
    "a", "an", "the", "this", "that", "these", "those", "my", "your", "his",
    "her", "its", "their", "our",
}
BOUNDARY_TRAILING_FUNCTION_WORDS = ARTICLES_DETERMINERS | {
    "and", "or", "but", "because", "so", "if", "when", "while", "with", "without",
    "of", "to", "for", "from", "in", "on", "at", "by", "as", "than", "into",
    "about", "like", "has", "have", "had", "having", "there", "here",
}
COMMON_PRE_NOMINAL_WORDS = {
    "right", "left", "big", "bigger", "small", "smaller", "little", "long",
    "longer", "short", "wide", "wider", "skinny", "skinnier", "straight",
    "flat", "curvy", "curved", "slanted", "equal", "same", "middle", "top",
    "bottom", "thin", "thick", "tiny", "large", "medium", "different",
    "similar", "perfect", "crooked", "round",
}
BOUNDARY_TRAILING_AUXILIARIES = {
    "i", "you", "we", "they", "he", "she", "it", "there", "that", "this",
    "i'm", "im", "i’d", "i'd", "i’ll", "i'll", "you're", "you’re", "it's",
    "it’s", "that's", "that’s", "there's", "there’s", "would", "could",
    "should", "will", "can", "can't", "don’t", "don't", "doesn’t", "doesn't",
    "is", "are", "was", "were", "be", "been", "being",
}
BOUNDARY_LEADING_CONTINUATIONS = {
    "gonna", "going", "wanna", "want", "need", "have", "has", "had", "would",
    "could", "should", "can", "be", "been", "being", "same", "right", "left",
    "triangle", "triangles", "square", "squares", "piece", "pieces", "side",
    "sides", "one", "ones", "thing", "things", "more", "too", "very",
    "really", "like", "not", "bigger", "smaller", "different", "similar",
}
BOUNDARY_TRAILING_ADVERBS = {
    "probably", "maybe", "just", "really", "very", "kind", "kinda", "sort",
    "sorta", "almost", "also", "still",
}
TASK_NOUNS = {
    "triangle", "triangles", "square", "squares", "rectangle", "rectangles",
    "circle", "circles", "shape", "shapes", "piece", "pieces", "line", "lines",
    "angle", "angles", "point", "points", "corner", "corners", "edge", "edges",
    "side", "sides", "crescent", "crescents", "trapezoid", "trapezoids",
    "top", "bottom", "front", "back",
}
KNOWN_TRUNCATED_FRAGMENTS = {
    "pa", "indentatio", "uninte", "t", "thi", "prob", "sec", "rectangl",
    "on", "th", "sam",
}


class SimilarityModel:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        if model_name.lower() == "lexical":
            return
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(model_name)
        except Exception as exc:  # pragma: no cover - depends on local model setup
            print(f"Warning: could not load embedding model '{model_name}': {exc}")
            print("Falling back to lexical overlap similarity.")

    def similarity(self, left: str, right: str) -> float:
        if not left.strip() or not right.strip():
            return 0.0
        if self.model is None:
            return lexical_similarity(left, right)
        vectors = self.model.encode([left, right], convert_to_numpy=True)
        denom = np.linalg.norm(vectors[0]) * np.linalg.norm(vectors[1])
        if denom == 0:
            return 0.0
        cosine = float(np.dot(vectors[0], vectors[1]) / denom)
        return max(0.0, min(1.0, (cosine + 1.0) / 2.0))


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def detokenize(tokens: Iterable[str]) -> str:
    text = " ".join(token for token in tokens if token != EVENT_TOKEN)
    return re.sub(r"\s+([.!?;:,])", r"\1", text).strip()


def lexical_similarity(left: str, right: str) -> float:
    left_words = {t.lower() for t in tokenize(left) if t.isalnum()}
    right_words = {t.lower() for t in tokenize(right) if t.isalnum()}
    if not left_words or not right_words:
        return 0.0
    jaccard = len(left_words & right_words) / len(left_words | right_words)
    left_norm = " ".join(sorted(left_words))
    right_norm = " ".join(sorted(right_words))
    sequence = SequenceMatcher(None, left_norm, right_norm).ratio()
    return max(jaccard, 0.35 * sequence)


def density(tokens: list[str], lexicon: set[str]) -> float:
    words = [t.lower() for t in tokens if re.search(r"[A-Za-z0-9]", t)]
    if not words:
        return 0.0
    return sum(1 for word in words if word in lexicon) / len(words)


def process_signal(tokens: list[str], lexicon: set[str]) -> float:
    return min(1.0, density(tokens, lexicon) * 4.0)


def thresholded_drop(drop: float, floor: float, ceiling: float) -> float:
    if ceiling <= floor:
        return max(0.0, min(1.0, drop))
    return max(0.0, min(1.0, (drop - floor) / (ceiling - floor)))


def cue_shift(left_tokens: list[str], right_tokens: list[str], lexicon: set[str]) -> float:
    left = {token.lower() for token in left_tokens if token.lower() in lexicon}
    right = {token.lower() for token in right_tokens if token.lower() in lexicon}
    if not left and not right:
        return 0.0
    if not left or not right:
        return min(0.4, (len(left) + len(right)) / 4.0)
    cue_volume = min(1.0, (len(left) + len(right)) / 4.0)
    return (1.0 - (len(left & right) / len(left | right))) * cue_volume


SEGMENT_SIGNAL_COLUMNS = {
    "cognitive": "cognitive_transition",
    "metacognitive": "metacognitive_reset",
    "affective": "affective_friction",
    "structural": "structural_break",
}


def classify_segment(evidence: list[dict]) -> tuple[dict[str, float], str, float]:
    if not evidence:
        return {name: math.nan for name in SEGMENT_SIGNAL_COLUMNS}, "low_signal", math.nan

    scores = {
        name: float(np.mean([row[column] for row in evidence]))
        for name, column in SEGMENT_SIGNAL_COLUMNS.items()
    }
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_name, best_score = ordered[0]
    second_score = ordered[1][1] if len(ordered) > 1 else 0.0
    confidence = max(0.0, best_score - second_score)

    active = [name for name, score in ordered if score >= 0.30]
    if best_score < 0.20:
        segment_type = "low_signal"
    elif len(active) >= 2 and confidence < 0.10:
        segment_type = "mixed_" + "_".join(active[:2])
    else:
        segment_type = best_name

    return scores, segment_type, confidence


def clean_transcript_artifacts(text: object) -> str:
    """Repair common transcript encoding artifacts without changing normal text."""
    if pd.isna(text):
        return ""
    cleaned = str(text).translate(CONTROL_TRANSLATION)
    cleaned = cleaned.replace("Â", "")
    cleaned = cleaned.replace("â€™", "'").replace("â€˜", "'")
    cleaned = cleaned.replace("â€œ", '"').replace("â€�", '"')
    cleaned = cleaned.replace("â€¦", "...").replace("â€“", " - ").replace("â€”", " - ")

    # This file contains sequences like "Thatï¿½ï¿½ï¿½s". The original exact
    # punctuation is already lost, but the linguistic repair is usually clear.
    repl = MOJIBAKE_REPLACEMENT
    cleaned = re.sub(f"(?:{repl})+(s|t|d|re|m|ll|ve)\\b", r"'\1", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(f"(?:{repl})+(cause|Cause|em)\\b", r"'\1", cleaned)
    cleaned = re.sub(f"\\bI(?:{repl})+(m|d|ll|ve)\\b", r"I'\1", cleaned, flags=re.IGNORECASE)
    cleaned = MOJIBAKE_REPLACEMENT_RE.sub(" - ", cleaned)
    cleaned = SPEAKER_LABEL_RE.sub("", cleaned)
    cleaned = re.sub(r"\s+-\s+-\s+", " - ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def count_encoding_artifacts(text: object) -> int:
    if pd.isna(text):
        return 0
    s = str(text)
    return (
        len(MOJIBAKE_REPLACEMENT_RE.findall(s))
        + len(SPEAKER_LABEL_RE.findall(s))
        + sum(s.count(ch) for ch in ["\x14", "\x19", "\x1c", "\x1d", "\xfd", "ý", "Â"])
        + sum(s.count(seq) for seq in ["â€™", "â€˜", "â€œ", "â€�", "â€¦", "â€“", "â€”"])
    )


def clean_dataframe_text(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    cleaned = df.copy()
    artifact_count = 0
    for col in cleaned.columns:
        if is_object_dtype(cleaned[col]) or is_string_dtype(cleaned[col]):
            artifact_count += int(cleaned[col].map(count_encoding_artifacts).sum())
            cleaned[col] = cleaned[col].map(clean_transcript_artifacts)
    return cleaned, artifact_count


def normalize_transcript(text: object, clean_text: bool = True) -> str:
    if pd.isna(text):
        return ""
    text = clean_transcript_artifacts(text) if clean_text else str(text)
    text = SPEAKER_LABEL_RE.sub("", text)
    text = text.replace(" | ", f" {EVENT_TOKEN} ")
    return " ".join(text.split())


def source_truncation_flag(text: str) -> bool:
    tokens = tokenize(text)
    if not tokens:
        return False
    last_word = next((token for token in reversed(tokens) if re.search(r"[A-Za-z0-9]", token)), "")
    if not last_word:
        return False
    lower = last_word.lower()
    common_short_words = {"i", "a", "an", "to", "of", "in", "on", "it", "is", "no"}
    if not text.rstrip()[-1:].isalnum():
        return False
    return lower in KNOWN_TRUNCATED_FRAGMENTS or (len(lower) <= 4 and lower not in common_short_words)


def prohibited_boundary_reason(left_token: str, right_token: str) -> str:
    left = left_token.lower()
    right = right_token.lower()
    if left_token == EVENT_TOKEN or right_token == EVENT_TOKEN:
        return ""
    if right_token in {".", ",", ":", ";", "!", "?"}:
        return "before_punctuation"
    if left in BOUNDARY_TRAILING_FUNCTION_WORDS:
        return "after_function_word"
    if left in BOUNDARY_TRAILING_AUXILIARIES:
        return "after_auxiliary_or_copula"
    if left in BOUNDARY_TRAILING_AUXILIARIES and right in BOUNDARY_LEADING_CONTINUATIONS:
        return "inside_verb_or_noun_phrase"
    if left in COMMON_PRE_NOMINAL_WORDS and right in TASK_NOUNS:
        return "inside_noun_phrase"
    if right in {"shaped", "shape"} and (len(left) == 1 or left in COMMON_PRE_NOMINAL_WORDS):
        return "inside_compound_word"
    if left in BOUNDARY_TRAILING_ADVERBS and right in BOUNDARY_LEADING_CONTINUATIONS:
        return "inside_adverbial_phrase"
    if right in {"but", "and", "or", "because", "so"} and left not in {".", "?", "!", ";", ":"}:
        return "before_connector"
    if left in {"not", "too", "more", "very", "really"}:
        return "after_modifier"
    return ""


SPACY_MODEL_CACHE = {}
SPACY_LOAD_WARNED: set[str] = set()


def _spacy_align_key(text: str) -> str:
    text = text.lower().replace("’", "'")
    if text == EVENT_TOKEN.lower():
        return ""
    return re.sub(r"^\W+|\W+$", "", text)


def _load_spacy_model(model_name: str):
    if model_name in SPACY_MODEL_CACHE:
        return SPACY_MODEL_CACHE[model_name]
    try:
        import spacy

        nlp = spacy.load(model_name)
        SPACY_MODEL_CACHE[model_name] = nlp
        return nlp
    except Exception as exc:  # pragma: no cover - depends on local spaCy setup
        if model_name not in SPACY_LOAD_WARNED:
            print(
                f"Warning: spaCy legality filter unavailable for model "
                f"'{model_name}': {exc}. Falling back to rule-based legality."
            )
            SPACY_LOAD_WARNED.add(model_name)
        SPACY_MODEL_CACHE[model_name] = None
        return None


class SpacyLegalityAnalyzer:
    """Extract parser-assisted legality features without replacing DAPS scoring."""

    ILLEGAL_DEPS = {"det", "amod", "compound", "nummod", "poss", "aux", "auxpass", "cop", "case", "mark"}
    NOMINAL_POS = {"NOUN", "PROPN", "PRON", "NUM"}
    HEAD_CONTINUATION_POS = {"NOUN", "PROPN", "ADJ", "VERB", "AUX", "PRON", "NUM"}
    DISCOURSE_CUES = {
        "actually", "but", "so", "then", "next", "okay", "wait", "because",
        "however", "now", "also", "first", "again",
    }

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.nlp = _load_spacy_model(model_name)

    @property
    def available(self) -> bool:
        return self.nlp is not None

    def analyze(self, tokens: list[str]) -> dict[int, dict]:
        if self.nlp is None:
            return {}

        text = detokenize(tokens)
        if not text:
            return {}

        doc = self.nlp(text)
        doc_tokens = [token for token in doc if not token.is_space]
        mapping = self._align_tokens(tokens, doc_tokens)
        noun_chunk_spans = self._noun_chunk_spans(doc)
        features = {}

        for gap in range(1, len(tokens)):
            left_doc_index = mapping.get(gap - 1)
            right_doc_index = mapping.get(gap)
            if left_doc_index is None or right_doc_index is None:
                continue
            if left_doc_index >= len(doc_tokens) or right_doc_index >= len(doc_tokens):
                continue

            left = doc_tokens[left_doc_index]
            right = doc_tokens[right_doc_index]
            same_sentence = left.sent.start == right.sent.start and left.sent.end == right.sent.end
            sentence_boundary = not same_sentence or bool(left.is_sent_end)
            inside_noun_chunk = any(
                start <= left.i and right.i < end
                for start, end in noun_chunk_spans
            )
            reasons = []
            legality = "legal"

            if inside_noun_chunk:
                reasons.append("inside_noun_chunk")
            if left.dep_ in self.ILLEGAL_DEPS and right.pos_ in self.HEAD_CONTINUATION_POS:
                reasons.append(f"left_dep_{left.dep_}_needs_head")
            if left.pos_ == "AUX" or left.dep_ in {"aux", "auxpass", "cop"}:
                if right.pos_ in {"VERB", "AUX", "ADJ", "NOUN", "PRON", "ADV"}:
                    reasons.append("inside_auxiliary_or_copula_phrase")
            if left.pos_ == "ADP" and right.pos_ in {"DET", "ADJ", "NOUN", "PROPN", "PRON", "NUM"}:
                reasons.append("inside_prepositional_phrase")
            if left.pos_ in {"ADJ", "ADV"} and right.pos_ in {"ADJ", "NOUN", "PROPN", "VERB", "AUX"}:
                reasons.append("inside_modifier_head_phrase")
            if right.is_punct:
                reasons.append("before_punctuation")

            if reasons:
                legality = "illegal"
            elif same_sentence and not sentence_boundary and right.lower_ not in self.DISCOURSE_CUES:
                if left.pos_ in {"VERB", "NOUN", "PROPN", "ADJ", "ADV"} and right.pos_ in self.NOMINAL_POS | {"ADJ", "ADV"}:
                    legality = "semi_legal"
                    reasons.append("within_sentence_noncue_gap")

            features[gap] = {
                "spacy_legality": legality,
                "spacy_legality_reason": ";".join(reasons),
                "spacy_left_pos": left.pos_,
                "spacy_right_pos": right.pos_,
                "spacy_left_dep": left.dep_,
                "spacy_right_dep": right.dep_,
                "spacy_same_sentence": bool(same_sentence),
                "spacy_sentence_boundary": bool(sentence_boundary),
                "spacy_inside_noun_chunk": bool(inside_noun_chunk),
            }
        return features

    def _align_tokens(self, daps_tokens: list[str], doc_tokens: list) -> dict[int, int | None]:
        mapping: dict[int, int | None] = {}
        doc_index = 0
        for daps_index, token in enumerate(daps_tokens):
            if token == EVENT_TOKEN:
                mapping[daps_index] = None
                continue
            key = _spacy_align_key(token)
            if not key and token not in {".", "!", "?", ",", ":", ";"}:
                mapping[daps_index] = None
                continue

            matched = None
            search_limit = min(len(doc_tokens), doc_index + 4)
            for candidate_index in range(doc_index, search_limit):
                doc_key = _spacy_align_key(doc_tokens[candidate_index].text)
                if key == doc_key or token == doc_tokens[candidate_index].text:
                    matched = candidate_index
                    break
            mapping[daps_index] = matched
            if matched is not None:
                doc_index = matched + 1
        return mapping

    def _noun_chunk_spans(self, doc) -> list[tuple[int, int]]:
        try:
            return [(chunk.start, chunk.end) for chunk in doc.noun_chunks]
        except Exception:
            return []


class DAPSSegmenter:
    def __init__(
        self,
        config: DAPSConfig,
        similarity_model: SimilarityModel,
        vocabulary: SignalVocabulary | None = None,
    ):
        self.config = config
        self.similarity_model = similarity_model
        self.vocabulary = vocabulary or SignalVocabulary()
        legality_mode = str(self.config.legality_mode).lower()
        self.spacy_legality = (
            SpacyLegalityAnalyzer(self.config.spacy_model)
            if legality_mode in {"spacy", "hybrid"}
            else None
        )

    def score_boundaries(self, tokens: list[str]) -> list[dict]:
        rows = []
        n = len(tokens)
        legality_mode = str(self.config.legality_mode).lower()
        use_spacy = self.spacy_legality is not None and self.spacy_legality.available
        use_rule = legality_mode in {"rule", "hybrid"} or (legality_mode == "spacy" and not use_spacy)
        spacy_features = self.spacy_legality.analyze(tokens) if use_spacy else {}
        for gap in range(1, n):
            left_tokens = tokens[max(0, gap - self.config.context_width) : gap]
            right_tokens = tokens[gap : min(n, gap + self.config.context_width)]
            context_tokens = left_tokens + right_tokens
            left_text = detokenize(left_tokens)
            right_text = detokenize(right_tokens)

            semantic = self.similarity_model.similarity(left_text, right_text)

            semantic_drop = 1.0 - semantic
            semantic_drop_component = thresholded_drop(
                semantic_drop,
                self.config.cognitive_drop_floor,
                self.config.cognitive_drop_ceiling,
            )
            cognitive_cue_shift = cue_shift(left_tokens, right_tokens, self.vocabulary.cognitive)
            cognitive_cue_evidence = process_signal(context_tokens, self.vocabulary.cognitive)
            semantic_gate = self.config.cognitive_semantic_gate_floor + (
                (1.0 - self.config.cognitive_semantic_gate_floor) * cognitive_cue_evidence
            )
            cognitive_transition = (
                self.config.cognitive_semantic_weight * semantic_drop_component * semantic_gate
                + self.config.cognitive_cue_shift_weight * cognitive_cue_shift
            )
            metacognitive_reset = process_signal(
                right_tokens[:4] + left_tokens[-2:], self.vocabulary.metacognitive
            )
            affective_friction = process_signal(context_tokens, self.vocabulary.affective)
            structural_break = process_signal(right_tokens[:3] + left_tokens[-1:], self.vocabulary.structural)
            transition_pressure = (
                0.45 * cognitive_transition
                + 0.25 * metacognitive_reset
                + 0.15 * affective_friction
                + 0.15 * structural_break
            )
            task_density = density(context_tokens, self.vocabulary.task)
            raw_gravity = (semantic * (1.0 + self.config.alpha_task_density * task_density)) - (
                self.config.transition_weight * transition_pressure
            )

            rule_reason = prohibited_boundary_reason(tokens[gap - 1], tokens[gap]) if use_rule else ""
            spacy_row = spacy_features.get(gap, {})
            spacy_legality = str(spacy_row.get("spacy_legality", "") or "")
            spacy_reason = str(spacy_row.get("spacy_legality_reason", "") or "")
            illegal_reasons = []
            if rule_reason:
                illegal_reasons.append(f"rule:{rule_reason}")
            if spacy_legality == "illegal" and spacy_reason:
                illegal_reasons.append(f"spacy:{spacy_reason}")

            if illegal_reasons:
                boundary_legality = "illegal"
                semi_penalty = 0.0
            elif spacy_legality == "semi_legal":
                boundary_legality = "semi_legal"
                semi_penalty = self.config.semi_legal_penalty
            else:
                boundary_legality = "legal"
                semi_penalty = 0.0

            sentence_boundary = bool(spacy_row.get("spacy_sentence_boundary", False)) or tokens[gap - 1] in {
                ".", "?", "!", ";", ":",
            }
            sentence_boost = self.config.sentence_boundary_boost if sentence_boundary and boundary_legality != "illegal" else 0.0
            adjusted_gravity = raw_gravity + semi_penalty - sentence_boost
            legality_reason = ";".join([reason for reason in [*illegal_reasons, spacy_reason] if reason])

            rows.append(
                {
                    "gap": gap,
                    "left_token": tokens[gap - 1],
                    "right_token": tokens[gap],
                    "semantic_continuity": round(semantic, 4),
                    "task_density": round(task_density, 4),
                    "cognitive_transition": round(cognitive_transition, 4),
                    "semantic_drop_component": round(semantic_drop_component, 4),
                    "cognitive_cue_evidence": round(cognitive_cue_evidence, 4),
                    "cognitive_cue_shift": round(cognitive_cue_shift, 4),
                    "metacognitive_reset": round(metacognitive_reset, 4),
                    "affective_friction": round(affective_friction, 4),
                    "structural_break": round(structural_break, 4),
                    "transition_pressure": round(transition_pressure, 4),
                    "raw_semantic_gravity": round(raw_gravity, 4),
                    "semantic_gravity": round(adjusted_gravity, 4),
                    "boundary_legality": boundary_legality,
                    "legality_reason": legality_reason,
                    "legality_penalty": round(semi_penalty, 4),
                    "sentence_boundary_boost": round(sentence_boost, 4),
                    "rule_constraint_reason": rule_reason,
                    "spacy_legality": spacy_legality,
                    "spacy_legality_reason": spacy_reason,
                    "spacy_left_pos": spacy_row.get("spacy_left_pos", ""),
                    "spacy_right_pos": spacy_row.get("spacy_right_pos", ""),
                    "spacy_left_dep": spacy_row.get("spacy_left_dep", ""),
                    "spacy_right_dep": spacy_row.get("spacy_right_dep", ""),
                    "spacy_same_sentence": spacy_row.get("spacy_same_sentence", ""),
                    "spacy_sentence_boundary": spacy_row.get("spacy_sentence_boundary", ""),
                    "spacy_inside_noun_chunk": spacy_row.get("spacy_inside_noun_chunk", ""),
                    "local_threshold": math.nan,
                    "boundary_margin": math.nan,
                    "candidate_boundary": False,
                    "selected_boundary": False,
                    "forced_max_length_boundary": False,
                    "constraint_blocked": bool(illegal_reasons),
                    "constraint_reason": ";".join(illegal_reasons),
                    "review_flag": False,
                    "context": f"{left_text} <BOUNDARY> {right_text}",
                }
            )
        return rows

    def select_boundaries(self, rows: list[dict], token_count: int) -> list[int]:
        if not rows:
            return []

        gravity = np.array([row["semantic_gravity"] for row in rows], dtype=float)
        candidates = []
        for i, row in enumerate(rows):
            lo = max(0, i - self.config.local_radius)
            hi = min(len(gravity), i + self.config.local_radius + 1)
            neighborhood = np.delete(gravity[lo:hi], i - lo)
            if len(neighborhood) == 0:
                local_mean = gravity[i]
                local_std = 0.0
            else:
                local_mean = float(np.mean(neighborhood))
                local_std = float(np.std(neighborhood))
            threshold = local_mean - self.config.sensitivity * local_std
            left_ok = i == 0 or gravity[i] <= gravity[i - 1]
            right_ok = i == len(gravity) - 1 or gravity[i] <= gravity[i + 1]
            is_candidate = gravity[i] < threshold and left_ok and right_ok
            margin = threshold - gravity[i]
            constraint_reason = str(row.get("constraint_reason", "") or "")

            row["local_threshold"] = round(threshold, 4)
            row["boundary_margin"] = round(margin, 4)
            row["candidate_boundary"] = bool(is_candidate)
            row["constraint_reason"] = constraint_reason
            row["review_flag"] = bool(is_candidate and margin <= self.config.ambiguous_margin)
            if is_candidate:
                if constraint_reason:
                    row["constraint_blocked"] = True
                    continue
                candidates.append((row["gap"], gravity[i], i))

        candidates.sort(key=lambda item: item[1])
        selected = []
        for gap, _score, row_index in candidates:
            with_new = sorted(selected + [gap])
            spans = [with_new[0], *[b - a for a, b in zip(with_new, with_new[1:])], token_count - with_new[-1]]
            if min(spans) < self.config.min_segment_tokens:
                continue
            if any(abs(gap - existing) <= self.config.nms_radius for existing in selected):
                continue
            selected.append(gap)
            rows[row_index]["selected_boundary"] = True

        return sorted(selected)

    def apply_max_length_splits(self, selected: list[int], rows: list[dict], token_count: int) -> list[int]:
        selected_set = set(selected)
        while True:
            cuts = [0] + sorted(selected_set) + [token_count]
            long_span = next(
                ((start, end) for start, end in zip(cuts, cuts[1:]) if end - start > self.config.max_segment_tokens),
                None,
            )
            if long_span is None:
                break

            start, end = long_span
            min_gap = start + self.config.min_segment_tokens
            max_gap = end - self.config.min_segment_tokens
            if min_gap > max_gap:
                break

            candidates = [
                row for row in rows
                if min_gap <= row["gap"] <= max_gap
                and row["gap"] not in selected_set
                and not row.get("constraint_reason")
            ]
            if not candidates:
                candidates = [
                    row for row in rows
                    if min_gap <= row["gap"] <= max_gap
                    and row["gap"] not in selected_set
                    and row["right_token"] not in {".", ",", ":", ";", "!", "?"}
                    and row["left_token"].lower() not in BOUNDARY_TRAILING_FUNCTION_WORDS
                ]
            if not candidates:
                break

            def forced_score(row: dict) -> tuple[int, float, float]:
                left = str(row["left_token"]).lower()
                right = str(row["right_token"]).lower()
                punctuation_bonus = 0
                if left in {".", "?", "!", ";", ":"}:
                    punctuation_bonus = -3
                elif right in {"okay", "then", "but", "and", "wait", "actually", "so"}:
                    punctuation_bonus = -2
                center_penalty = abs(row["gap"] - ((start + end) / 2.0)) / max(1, end - start)
                return (punctuation_bonus, row["semantic_gravity"], center_penalty)

            chosen = min(candidates, key=forced_score)
            selected_set.add(chosen["gap"])
            chosen["selected_boundary"] = True
            chosen["forced_max_length_boundary"] = True

        return sorted(selected_set)

    def segment_tokens(self, tokens: list[str], offset: int = 0) -> tuple[list[dict], list[dict]]:
        if len(tokens) <= 1:
            return [], []
        boundary_rows = self.score_boundaries(tokens)
        boundaries = self.select_boundaries(boundary_rows, len(tokens))
        boundaries = self.apply_max_length_splits(boundaries, boundary_rows, len(tokens))
        cuts = [0] + boundaries + [len(tokens)]
        segment_rows = []
        for index, (start, end) in enumerate(zip(cuts, cuts[1:]), start=1):
            segment_tokens = tokens[start:end]
            segment_text = detokenize(segment_tokens)
            if not segment_text:
                continue
            evidence = [
                row
                for row in boundary_rows
                if start < row["gap"] < end or row["gap"] in {start, end}
            ]
            signal_scores, segment_type, segment_type_confidence = classify_segment(evidence)
            segment_rows.append(
                {
                    "Segment_Index": index,
                    "Token_Start": start + offset,
                    "Token_End": end + offset,
                    "Token_Count": end - start,
                    "Segment_Text": segment_text,
                    "Segment_Type": segment_type,
                    "Segment_Type_Confidence": round(segment_type_confidence, 4)
                    if not pd.isna(segment_type_confidence)
                    else math.nan,
                    "Mean_C_t": round(signal_scores["cognitive"], 4)
                    if not pd.isna(signal_scores["cognitive"])
                    else math.nan,
                    "Mean_M_t": round(signal_scores["metacognitive"], 4)
                    if not pd.isna(signal_scores["metacognitive"])
                    else math.nan,
                    "Mean_A_t": round(signal_scores["affective"], 4)
                    if not pd.isna(signal_scores["affective"])
                    else math.nan,
                    "Mean_R_t": round(signal_scores["structural"], 4)
                    if not pd.isna(signal_scores["structural"])
                    else math.nan,
                    "Source_Truncation_Flag": False,
                    "Mean_Semantic_Gravity": round(float(np.mean([r["semantic_gravity"] for r in evidence])), 4)
                    if evidence
                    else math.nan,
                    "Mean_Task_Density": round(float(np.mean([r["task_density"] for r in evidence])), 4)
                    if evidence
                    else math.nan,
                    "Mean_Transition_Pressure": round(
                        float(np.mean([r["transition_pressure"] for r in evidence])), 4
                    )
                    if evidence
                    else math.nan,
                }
            )
        for row in boundary_rows:
            row["gap"] += offset
        return segment_rows, boundary_rows

    def apply_segment_count_cap(self, segments: list[dict]) -> list[dict]:
        cap = self.config.max_segments_per_record
        if cap <= 0 or len(segments) <= cap:
            return segments

        merged = [dict(segment) for segment in segments]
        while len(merged) > cap:
            candidates = []
            for index in range(len(merged) - 1):
                left = merged[index]
                right = merged[index + 1]
                combined_tokens = int(left["Token_Count"]) + int(right["Token_Count"])
                if combined_tokens <= self.config.max_segment_tokens:
                    candidates.append((combined_tokens, index))
            if not candidates:
                break

            _combined_tokens, index = min(candidates)
            left = merged[index]
            right = merged[index + 1]
            combined_text = detokenize(tokenize(f"{left['Segment_Text']} {right['Segment_Text']}"))
            left["Token_End"] = right["Token_End"]
            left["Token_Count"] = int(left["Token_Count"]) + int(right["Token_Count"])
            left["Segment_Text"] = combined_text
            left["Merged_By_Max_Segment_Count"] = True
            left["Source_Truncation_Flag"] = bool(left.get("Source_Truncation_Flag") or right.get("Source_Truncation_Flag"))
            for metric in (
                "Mean_Semantic_Gravity",
                "Mean_Task_Density",
                "Mean_Transition_Pressure",
                "Mean_C_t",
                "Mean_M_t",
                "Mean_A_t",
                "Mean_R_t",
            ):
                values = [
                    value for value in (left.get(metric), right.get(metric))
                    if not pd.isna(value)
                ]
                left[metric] = round(float(np.mean(values)), 4) if values else math.nan
            merged_scores = {
                "cognitive": left.get("Mean_C_t", math.nan),
                "metacognitive": left.get("Mean_M_t", math.nan),
                "affective": left.get("Mean_A_t", math.nan),
                "structural": left.get("Mean_R_t", math.nan),
            }
            ordered = sorted(
                ((name, score) for name, score in merged_scores.items() if not pd.isna(score)),
                key=lambda item: item[1],
                reverse=True,
            )
            if ordered:
                best_name, best_score = ordered[0]
                second_score = ordered[1][1] if len(ordered) > 1 else 0.0
                confidence = max(0.0, best_score - second_score)
                active = [name for name, score in ordered if score >= 0.30]
                if best_score < 0.20:
                    left["Segment_Type"] = "low_signal"
                elif len(active) >= 2 and confidence < 0.10:
                    left["Segment_Type"] = "mixed_" + "_".join(active[:2])
                else:
                    left["Segment_Type"] = best_name
                left["Segment_Type_Confidence"] = round(confidence, 4)
            del merged[index + 1]

        for index, segment in enumerate(merged, start=1):
            segment["Segment_Index"] = index
            segment.setdefault("Merged_By_Max_Segment_Count", False)
        return merged

    def segment(self, text: str) -> tuple[list[dict], list[dict]]:
        tokens = tokenize(text)
        if len(tokens) <= 1:
            return [], []

        all_segments = []
        all_boundaries = []
        chunk = []
        offset = 0
        transcript_truncation = source_truncation_flag(text)

        def flush_chunk() -> None:
            nonlocal chunk, offset, all_segments, all_boundaries
            if chunk:
                segments, boundaries = self.segment_tokens(chunk, offset)
                all_segments.extend(segments)
                all_boundaries.extend(boundaries)
                offset += len(chunk)
                chunk = []

        for token in tokens:
            if token == EVENT_TOKEN:
                flush_chunk()
                offset += 1
                continue
            chunk.append(token)
        flush_chunk()

        for index, segment in enumerate(all_segments, start=1):
            segment["Segment_Index"] = index
            segment.setdefault("Merged_By_Max_Segment_Count", False)
            segment["Source_Truncation_Flag"] = bool(
                index == len(all_segments) and transcript_truncation
            )

        all_segments = self.apply_segment_count_cap(all_segments)
        if all_segments:
            all_segments[-1]["Source_Truncation_Flag"] = bool(transcript_truncation)

        return all_segments, all_boundaries


def read_table(path: Path, sheet_name: str | int | None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xlsm", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet_name or 0)
    if suffix == ".csv":
        for encoding in ("utf-8-sig", "utf-8", "windows-1252"):
            try:
                return pd.read_csv(path, encoding=encoding)
            except UnicodeDecodeError:
                continue
        return pd.read_csv(path, encoding="utf-8", encoding_errors="replace")
    raise ValueError(f"Unsupported input file type: {suffix}")


def infer_id_column(df: pd.DataFrame, requested: str | None) -> str | None:
    if requested:
        if requested not in df.columns:
            raise ValueError(f"ID column not found: {requested}")
        return requested
    for candidate in ("ID", "Id", "id", "Student_ID", "student_id", "Participant", "participant"):
        if candidate in df.columns:
            return candidate
    return None


def infer_text_columns(df: pd.DataFrame, requested: str | None, id_column: str | None) -> list[str]:
    if requested:
        columns = [col.strip() for col in requested.split(",") if col.strip()]
        missing = [col for col in columns if col not in df.columns]
        if missing:
            raise ValueError(f"Text column(s) not found: {', '.join(missing)}")
        return columns

    priority_patterns = [
        r"^Interviewee_Response$",
        r"^Transcript$",
        r"^Text$",
        r"^Content$",
        r"^Item\s*\d+$",
    ]
    selected = []
    for pattern in priority_patterns:
        selected.extend([col for col in df.columns if re.search(pattern, str(col), re.IGNORECASE)])
    selected = list(dict.fromkeys(selected))
    if selected:
        return selected

    excluded = {id_column} if id_column else set()
    excluded_patterns = re.compile(r"choice|choic|score|response\s*\d+|correct", re.IGNORECASE)
    text_like = []
    for col in df.columns:
        if col in excluded or excluded_patterns.search(str(col)):
            continue
        if is_object_dtype(df[col]) or is_string_dtype(df[col]):
            text_like.append(col)
    if not text_like:
        raise ValueError("Could not infer transcript columns. Use --text-columns.")
    return text_like


def run_excel_segmentation(args: argparse.Namespace) -> None:
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    df = read_table(input_path, args.sheet)
    if args.clean_text:
        df, artifact_count = clean_dataframe_text(df)
        print(f"Cleaned {artifact_count} encoding artifact(s) from text cells.")
    id_column = infer_id_column(df, args.id_column)
    text_columns = infer_text_columns(df, args.text_columns, id_column)

    segmenter = DAPSSegmenter(
        DAPSConfig(
            context_width=args.context_width,
            local_radius=args.local_radius,
            sensitivity=args.sensitivity,
            min_segment_tokens=args.min_segment_tokens,
            nms_radius=args.nms_radius,
            max_segment_tokens=args.max_segment_tokens,
            max_segments_per_record=args.max_segments_per_record,
            transition_weight=args.transition_weight,
            cognitive_drop_floor=args.cognitive_drop_floor,
            cognitive_drop_ceiling=args.cognitive_drop_ceiling,
            cognitive_semantic_weight=args.cognitive_semantic_weight,
            cognitive_cue_shift_weight=args.cognitive_cue_shift_weight,
            cognitive_semantic_gate_floor=args.cognitive_semantic_gate_floor,
            legality_mode=args.legality_mode,
            spacy_model=args.spacy_model,
            semi_legal_penalty=args.semi_legal_penalty,
            sentence_boundary_boost=args.sentence_boundary_boost,
        ),
        SimilarityModel(args.embedding_model),
        SignalVocabulary.from_texts(
            task=args.task_vocab,
            cognitive=args.cognitive_vocab,
            metacognitive=args.metacognitive_vocab,
            affective=args.affective_vocab,
            structural=args.structural_vocab,
        ),
    )

    all_segments = []
    all_boundaries = []
    for row_index, row in df.iterrows():
        record_id = row[id_column] if id_column else row_index + 1
        for source_column in text_columns:
            text = normalize_transcript(row[source_column], clean_text=args.clean_text)
            if not text:
                continue
            segments, boundaries = segmenter.segment(text)
            for segment in segments:
                all_segments.append(
                    {
                        "Record_ID": record_id,
                        "Source_Row": row_index + 2,
                        "Source_Column": source_column,
                        **segment,
                    }
                )
            for boundary in boundaries:
                all_boundaries.append(
                    {
                        "Record_ID": record_id,
                        "Source_Row": row_index + 2,
                        "Source_Column": source_column,
                        **boundary,
                    }
                )
            print(f"Processed {record_id} / {source_column}: {len(segments)} segment(s)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        pd.DataFrame(all_segments).to_excel(writer, sheet_name="segments", index=False)
        pd.DataFrame(all_boundaries).to_excel(writer, sheet_name="boundaries", index=False)

    print(f"Saved DAPS segmentation workbook: {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Apply DAPS segmentation to an Excel/CSV transcript table.")
    parser.add_argument("input", help="Path to .xlsx, .xls, .xlsm, or .csv file.")
    parser.add_argument("-o", "--output", default="../data/daps_segments.xlsx", help="Output .xlsx path.")
    parser.add_argument("--sheet", default=None, help="Excel sheet name or index. Defaults to the first sheet.")
    parser.add_argument("--id-column", default=None, help="Optional participant/record ID column.")
    parser.add_argument(
        "--text-columns",
        default=None,
        help="Comma-separated transcript columns. Defaults to Interviewee_Response, Transcript/Text, or Item 1..n.",
    )
    parser.add_argument("--embedding-model", default="all-mpnet-base-v2", help="SentenceTransformer model or 'lexical'.")
    parser.add_argument("--context-width", type=int, default=12, help="Tokens on each side of a candidate boundary.")
    parser.add_argument("--local-radius", type=int, default=6, help="Neighborhood radius for adaptive valley threshold.")
    parser.add_argument("--sensitivity", type=float, default=0.55, help="Higher values produce fewer boundaries.")
    parser.add_argument("--min-segment-tokens", type=int, default=12, help="Minimum tokens per accepted segment.")
    parser.add_argument("--max-segment-tokens", type=int, default=60, help="Force an additional legal split above this segment length.")
    parser.add_argument(
        "--max-segments-per-record",
        type=int,
        default=80,
        help="Merge adjacent short segments if one record exceeds this count. Use 0 to disable.",
    )
    parser.add_argument("--nms-radius", type=int, default=6, help="Suppress nearby candidate valleys within this radius.")
    parser.add_argument("--transition-weight", type=float, default=0.55, help="Weight applied to transition pressure.")
    parser.add_argument(
        "--cognitive-drop-floor",
        type=float,
        default=0.35,
        help="Semantic drop must exceed this floor before contributing to C_t.",
    )
    parser.add_argument(
        "--cognitive-drop-ceiling",
        type=float,
        default=0.80,
        help="Semantic drop at or above this value saturates the semantic component of C_t.",
    )
    parser.add_argument(
        "--cognitive-semantic-weight",
        type=float,
        default=0.45,
        help="Weight of thresholded semantic drop inside C_t.",
    )
    parser.add_argument(
        "--cognitive-cue-shift-weight",
        type=float,
        default=0.55,
        help="Weight of left/right cognitive cue shift inside C_t.",
    )
    parser.add_argument(
        "--cognitive-semantic-gate-floor",
        type=float,
        default=0.25,
        help="Minimum multiplier for semantic-drop contribution when cognitive cue evidence is absent.",
    )
    parser.add_argument(
        "--legality-mode",
        choices=["hybrid", "rule", "spacy", "off"],
        default="hybrid",
        help="Boundary legality filter: hybrid combines rule and spaCy parser features; rule uses lexical rules only.",
    )
    parser.add_argument(
        "--spacy-model",
        default="en_core_web_sm",
        help="spaCy English pipeline used for parser-assisted boundary legality features.",
    )
    parser.add_argument(
        "--semi-legal-penalty",
        type=float,
        default=0.18,
        help="Penalty added to G_t for semi-legal gaps, requiring stronger DAPS evidence.",
    )
    parser.add_argument(
        "--sentence-boundary-boost",
        type=float,
        default=0.08,
        help="Small reduction to G_t at parser/sentence boundaries, making legal sentence breaks easier to select.",
    )
    parser.add_argument("--task-vocab", default=None, help="Comma/space/newline separated task-density vocabulary.")
    parser.add_argument("--cognitive-vocab", default=None, help="Comma/space/newline separated cognitive-transition cue vocabulary.")
    parser.add_argument("--metacognitive-vocab", default=None, help="Comma/space/newline separated metacognitive-reset vocabulary.")
    parser.add_argument("--affective-vocab", default=None, help="Comma/space/newline separated affective-friction vocabulary.")
    parser.add_argument("--structural-vocab", default=None, help="Comma/space/newline separated rhetorical/structural-break vocabulary.")
    parser.add_argument(
        "--clean-text",
        dest="clean_text",
        action="store_true",
        default=True,
        help="Automatically clean common encoding artifacts before segmentation. Enabled by default.",
    )
    parser.add_argument(
        "--no-clean-text",
        dest="clean_text",
        action="store_false",
        help="Disable automatic encoding-artifact cleanup.",
    )
    return parser


if __name__ == "__main__":
    run_excel_segmentation(build_parser().parse_args())
