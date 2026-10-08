"""
Tests for tracks/reasoning/prompt_template.py — Task 5.

Covers:
  - All ten Privacy_Category names present in every prompt variant
  - Deterministic output (same inputs → same prompt)
  - Context-free prompt contains no PATIENT CONTEXT section
  - Correct-context and shuffled-context prompts contain PATIENT CONTEXT section
  - get_prompt_hash returns a 16-character hex string
  - verify_no_physician_labels returns True for all three prompt variants
  - verify_no_physician_labels returns False when forbidden terms are injected
"""

from __future__ import annotations

import hashlib

import pytest

from tracks.reasoning.prompt_template import (
    CATEGORY_NAMES,
    PRIVACY_CATEGORIES,
    PromptTemplate,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def template() -> PromptTemplate:
    return PromptTemplate()


SAMPLE_ITEM = "Patient takes lithium 300 mg twice daily for bipolar disorder."
SAMPLE_CONTEXT = "[SUMMARY]\nAdult patient with chronic conditions.\n\n[MEDICAL_HISTORY]\nHypertension."
SAMPLE_SHUFFLED_CONTEXT = "[SUMMARY]\nDifferent patient context.\n\n[MEDICAL_HISTORY]\nDiabetes."


# ---------------------------------------------------------------------------
# 5.1 — All ten Privacy_Category definitions present
# ---------------------------------------------------------------------------

class TestPrivacyCategoryDefinitions:
    def test_exactly_ten_categories(self):
        assert len(PRIVACY_CATEGORIES) == 10

    def test_category_names_match_glossary(self):
        expected = {
            "behavioral_health",
            "diagnoses",
            "disabilities",
            "infectious_diseases",
            "genetics",
            "medications",
            "sexual_reproductive_health",
            "social_determinants_of_health",
            "violence",
            "other",
        }
        assert set(PRIVACY_CATEGORIES.keys()) == expected

    def test_category_names_list_length(self):
        assert len(CATEGORY_NAMES) == 10

    def test_category_names_list_matches_dict_keys(self):
        assert CATEGORY_NAMES == list(PRIVACY_CATEGORIES.keys())

    def test_all_definitions_non_empty(self):
        for name, definition in PRIVACY_CATEGORIES.items():
            assert definition.strip(), f"Definition for '{name}' is empty"


# ---------------------------------------------------------------------------
# 5.2 — build_context_free_prompt
# ---------------------------------------------------------------------------

class TestContextFreePrompt:
    def test_contains_item_text(self, template):
        prompt = template.build_context_free_prompt(SAMPLE_ITEM)
        assert SAMPLE_ITEM in prompt

    def test_contains_all_ten_category_names(self, template):
        prompt = template.build_context_free_prompt(SAMPLE_ITEM)
        for name in CATEGORY_NAMES:
            assert name in prompt, f"Category '{name}' missing from context-free prompt"

    def test_no_patient_context_section(self, template):
        prompt = template.build_context_free_prompt(SAMPLE_ITEM)
        assert "PATIENT CONTEXT:" not in prompt

    def test_contains_json_format_instruction(self, template):
        prompt = template.build_context_free_prompt(SAMPLE_ITEM)
        assert "JSON" in prompt

    def test_deterministic(self, template):
        p1 = template.build_context_free_prompt(SAMPLE_ITEM)
        p2 = template.build_context_free_prompt(SAMPLE_ITEM)
        assert p1 == p2

    def test_different_items_produce_different_prompts(self, template):
        p1 = template.build_context_free_prompt("item A")
        p2 = template.build_context_free_prompt("item B")
        assert p1 != p2


# ---------------------------------------------------------------------------
# 5.3 — build_correct_context_prompt
# ---------------------------------------------------------------------------

class TestCorrectContextPrompt:
    def test_contains_item_text(self, template):
        prompt = template.build_correct_context_prompt(SAMPLE_ITEM, SAMPLE_CONTEXT)
        assert SAMPLE_ITEM in prompt

    def test_contains_patient_context_section(self, template):
        prompt = template.build_correct_context_prompt(SAMPLE_ITEM, SAMPLE_CONTEXT)
        assert "PATIENT CONTEXT:" in prompt

    def test_contains_context_text(self, template):
        prompt = template.build_correct_context_prompt(SAMPLE_ITEM, SAMPLE_CONTEXT)
        assert SAMPLE_CONTEXT in prompt

    def test_contains_all_ten_category_names(self, template):
        prompt = template.build_correct_context_prompt(SAMPLE_ITEM, SAMPLE_CONTEXT)
        for name in CATEGORY_NAMES:
            assert name in prompt, f"Category '{name}' missing from correct-context prompt"

    def test_deterministic(self, template):
        p1 = template.build_correct_context_prompt(SAMPLE_ITEM, SAMPLE_CONTEXT)
        p2 = template.build_correct_context_prompt(SAMPLE_ITEM, SAMPLE_CONTEXT)
        assert p1 == p2

    def test_differs_from_context_free(self, template):
        cf = template.build_context_free_prompt(SAMPLE_ITEM)
        cc = template.build_correct_context_prompt(SAMPLE_ITEM, SAMPLE_CONTEXT)
        assert cf != cc


# ---------------------------------------------------------------------------
# 5.4 — build_shuffled_context_prompt
# ---------------------------------------------------------------------------

class TestShuffledContextPrompt:
    def test_contains_item_text(self, template):
        prompt = template.build_shuffled_context_prompt(SAMPLE_ITEM, SAMPLE_SHUFFLED_CONTEXT)
        assert SAMPLE_ITEM in prompt

    def test_contains_patient_context_section(self, template):
        prompt = template.build_shuffled_context_prompt(SAMPLE_ITEM, SAMPLE_SHUFFLED_CONTEXT)
        assert "PATIENT CONTEXT:" in prompt

    def test_contains_shuffled_context_text(self, template):
        prompt = template.build_shuffled_context_prompt(SAMPLE_ITEM, SAMPLE_SHUFFLED_CONTEXT)
        assert SAMPLE_SHUFFLED_CONTEXT in prompt

    def test_contains_all_ten_category_names(self, template):
        prompt = template.build_shuffled_context_prompt(SAMPLE_ITEM, SAMPLE_SHUFFLED_CONTEXT)
        for name in CATEGORY_NAMES:
            assert name in prompt, f"Category '{name}' missing from shuffled-context prompt"

    def test_deterministic(self, template):
        p1 = template.build_shuffled_context_prompt(SAMPLE_ITEM, SAMPLE_SHUFFLED_CONTEXT)
        p2 = template.build_shuffled_context_prompt(SAMPLE_ITEM, SAMPLE_SHUFFLED_CONTEXT)
        assert p1 == p2

    def test_differs_from_correct_context_when_context_differs(self, template):
        cc = template.build_correct_context_prompt(SAMPLE_ITEM, SAMPLE_CONTEXT)
        sc = template.build_shuffled_context_prompt(SAMPLE_ITEM, SAMPLE_SHUFFLED_CONTEXT)
        assert cc != sc

    def test_same_structure_as_correct_context_when_same_context(self, template):
        """Structurally identical to correct-context when the same context text is used."""
        cc = template.build_correct_context_prompt(SAMPLE_ITEM, SAMPLE_CONTEXT)
        sc = template.build_shuffled_context_prompt(SAMPLE_ITEM, SAMPLE_CONTEXT)
        assert cc == sc


# ---------------------------------------------------------------------------
# 5.5 — get_prompt_hash
# ---------------------------------------------------------------------------

class TestGetPromptHash:
    def test_returns_16_characters(self, template):
        prompt = template.build_context_free_prompt(SAMPLE_ITEM)
        h = PromptTemplate.get_prompt_hash(prompt)
        assert len(h) == 16

    def test_returns_hex_string(self, template):
        prompt = template.build_context_free_prompt(SAMPLE_ITEM)
        h = PromptTemplate.get_prompt_hash(prompt)
        int(h, 16)  # raises ValueError if not valid hex

    def test_deterministic(self, template):
        prompt = template.build_context_free_prompt(SAMPLE_ITEM)
        assert PromptTemplate.get_prompt_hash(prompt) == PromptTemplate.get_prompt_hash(prompt)

    def test_matches_sha256_prefix(self):
        text = "hello world"
        expected = hashlib.sha256(text.encode()).hexdigest()[:16]
        assert PromptTemplate.get_prompt_hash(text) == expected

    def test_different_prompts_produce_different_hashes(self, template):
        p1 = template.build_context_free_prompt("item A")
        p2 = template.build_context_free_prompt("item B")
        assert PromptTemplate.get_prompt_hash(p1) != PromptTemplate.get_prompt_hash(p2)


# ---------------------------------------------------------------------------
# 5.6 — verify_no_physician_labels
# ---------------------------------------------------------------------------

class TestVerifyNoPhysicianLabels:
    """No physician labels or ground truth in any prompt variant."""

    def test_context_free_prompt_is_clean(self, template):
        prompt = template.build_context_free_prompt(SAMPLE_ITEM)
        assert PromptTemplate.verify_no_physician_labels(prompt) is True

    def test_correct_context_prompt_is_clean(self, template):
        prompt = template.build_correct_context_prompt(SAMPLE_ITEM, SAMPLE_CONTEXT)
        assert PromptTemplate.verify_no_physician_labels(prompt) is True

    def test_shuffled_context_prompt_is_clean(self, template):
        prompt = template.build_shuffled_context_prompt(SAMPLE_ITEM, SAMPLE_SHUFFLED_CONTEXT)
        assert PromptTemplate.verify_no_physician_labels(prompt) is True

    @pytest.mark.parametrize("forbidden_term", [
        "Physician_Survey_Consensus",
        "Physician_Interview_Consensus",
        "Delta_Physician",
        "Delta_LLM",
        "ground truth",
        "ground_truth",
    ])
    def test_detects_forbidden_terms(self, forbidden_term):
        contaminated = f"Some prompt text. {forbidden_term} is here."
        assert PromptTemplate.verify_no_physician_labels(contaminated) is False

    @pytest.mark.parametrize("forbidden_term", [
        "PHYSICIAN_SURVEY_CONSENSUS",
        "physician_interview_consensus",
        "DELTA_PHYSICIAN",
        "Ground Truth",
    ])
    def test_case_insensitive_detection(self, forbidden_term):
        contaminated = f"Some prompt text. {forbidden_term} is here."
        assert PromptTemplate.verify_no_physician_labels(contaminated) is False

    def test_clean_prompt_returns_true(self):
        clean = "You are a clinical privacy expert. Classify this item."
        assert PromptTemplate.verify_no_physician_labels(clean) is True

    def test_all_three_variants_pass_for_various_items(self, template):
        items = [
            "Patient has HIV.",
            "Prescribed metformin 500mg.",
            "History of domestic violence.",
            "Genetic testing for BRCA1.",
        ]
        for item in items:
            cf = template.build_context_free_prompt(item)
            cc = template.build_correct_context_prompt(item, SAMPLE_CONTEXT)
            sc = template.build_shuffled_context_prompt(item, SAMPLE_SHUFFLED_CONTEXT)
            assert PromptTemplate.verify_no_physician_labels(cf), f"CF prompt contaminated for item: {item}"
            assert PromptTemplate.verify_no_physician_labels(cc), f"CC prompt contaminated for item: {item}"
            assert PromptTemplate.verify_no_physician_labels(sc), f"SC prompt contaminated for item: {item}"
