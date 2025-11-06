"""
###########################################################
NOTICE
Code in this script is licensed and distributed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
You must provide attribution. You may not use the code for commercial purposes. Any derivatives must be shared under the same license.

Copyright (c) 2025 Authors of [Data-Efficient Adaptation and a Novel Evaluation Method for Aspect-based Sentiment Analysis](https://arxiv.org/abs/2511.03034).
###########################################################
"""

import re
import ast
import string
from typing import Optional, Dict, Tuple, List, Union, Callable
from tqdm.notebook import tqdm
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from rouge_score import rouge_scorer
from collections import defaultdict

try:
    from IPython.display import display
except ImportError:
    display = print  # Fallback if IPython is not available


class ABSAEvaluator:
    """
    Evaluator for aspect-based sentiment extraction tasks with support for:
    1. Opinion extraction (OE)
    2. Aspect-opinion pair extraction (AOPE)
    3. Aspect-opinion pair extraction and classification (AOC)
    4. Aspect-opinion-sentiment triplet extraction (ASTE)
    5. Aspect-opinion-category-sentiment quadruplet extraction (ASQE)

    Applies FTS-OBP method to evaluate unit-level (within-entry pair/triplet/quadruplet) matches.
    Please view the paper for more details and process illustration.
    """

    """# ===========================================================================
    # Initialization and Configuration
    # ==========================================================================="""

    def __init__(self,
            prompt_dict: Optional[Dict[str, dict]] = None,
            category_delimiter: str = "-",
            weights: Optional[Dict[str, int]] = None,
            thresholds: Optional[Union[Dict, List, Callable]] = None,
            stopword_list: Optional[List[str]] = None,
            equal_weights: bool = True,
            partial_category_score: float = 0.3,
            allow_partial_category_for_unit_match: bool = False,
            implicit_aspect_tokens: Optional[List[str]] = ['null']):

        """Initialize the ABSA evaluator.
        Args:
            prompt_dict (Optional[Dict[str, dict]]): dictionary containing prompt version name (key) and the prompt string
            category_delimiter (str): delimiter of main and sub category labels (e.g. '-' for 'Course - Content')
            thresholds: governs whether an aspect or opinion gold-pred pair's similarity score (at pair/triplet/quadruplet level) is considered a match
                - Dict (fixed): {"high_match": 0.8, "partial_match": 0.4}
                - List of tuples (adaptive): [(max_length, {"high_match": ..., "partial_match": ...}), ...]
                - Callable (custom): function(gold_length) -> {"high_match": ..., "partial_match": ...}
            partial_category_score (float): score given to only main-category match (also used as category match threshold if allow_partial_category_for_unit_match = True).
            allow_partial_category_for_unit_match (bool): For multi-level category labels: If True, main-category-only matches count as passing for unit_match. If False, require exact full match.
            implicit_aspect_tokens (Optional[List[str]]): list of words indicate null extraction (implicit aspect), that is outside the original text but should be considered a legitimate extraction in flexible_text_similarity.
        """

        # Default to equal weights if not provided
        self.prompt_dict = prompt_dict
        self.component_order = ["aspect", "opinion", "category", "sentiment"]
        self.category_delimiter = category_delimiter
        self.implicit_aspect_tokens = implicit_aspect_tokens
        self.scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)

        # Category matching configuration
        self.partial_category_score = partial_category_score  # Score given for main-category-only match
        self.allow_partial_category_for_unit_match = allow_partial_category_for_unit_match

        if weights is None:
            self.weights = {} if equal_weights else {
                "aspect": 0.3,
                "opinion": 0.3,
                "category": 0.2,
                "sentiment": 0.2
            }
        else:
            self.weights = weights

        self.stopwords = set(stopword_list) if stopword_list else {"a", "an", "the", "is", "are", "was", "were", "be",
                                                                   "to", "of", "and", "in", "this", "that", "have",
                                                                   "it", "very", "really", "extremely", "super",
                                                                   "absolutely", "definitely"}

        # Detect threshold type and set defaults
        if thresholds is None:
            # Default adaptive thresholds
            self.threshold_type = 'adaptive'
            self.thresholds = {
                2: {"high_match": 0.50, "partial_match": 0.30},  # Accept off-by-1
                4: {"high_match": 0.60, "partial_match": 0.40},  # Accept off-by-1 to off-by-2
                None: {"high_match": 0.70, "partial_match": 0.50},  # Longer phrases need tighter match
                "overall_weighted": 0.7
            }
        elif any(isinstance(k, int) for k in thresholds.keys()) or None in thresholds:
            # Has integer keys or None -> adaptive
            self.threshold_type = 'adaptive'
            self.thresholds = thresholds
        elif 'high_match' in thresholds and 'partial_match' in thresholds:
            # Has threshold keys directly -> fixed
            self.threshold_type = 'fixed'
            self.thresholds = thresholds
        else:
            raise ValueError("Invalid threshold format")

    def get_thresholds(self, gold_length: int = None) -> Dict[str, float]:
        """Get thresholds based on gold length (if adaptive) or return fixed thresholds."""
        if self.threshold_type == 'fixed':
            return self.thresholds
        elif gold_length is None:
            return self.thresholds[None]
        else:
            # Adaptive: find appropriate tier - only sort integer keys
            sorted_lengths = sorted([k for k in self.thresholds.keys() if k is not None and isinstance(k, int)])
            for max_length in sorted_lengths:
                if gold_length <= max_length:
                    return self.thresholds[max_length]
            # Return default (None key)
            return self.thresholds[None]

    def get_gold_length(self, gold_text: str) -> int:
        """
        Calculate gold length after stopword removal (used for adaptive thresholding).
        Returns:
            int: Number of tokens after removing stopwords (minimum 1)
        """
        gold_tokens = [t for t in gold_text.strip().lower().split() if t not in self.stopwords]
        return len(gold_tokens) if gold_tokens else 1

    def get_components_for_task(self, task_type):
        """Get components relevant for a specific task type in the specified order."""
        if task_type == "ASQE":
            components = ["aspect", "opinion", "category", "sentiment"]
        elif task_type == "ASTE":
            components = ["aspect", "opinion", "sentiment"]
        elif task_type == "AOC":
            components = ["aspect", "opinion", "category"]
        elif task_type == "AOPE":
            components = ["aspect", "opinion"]
        elif task_type == "OE":
            components = ["opinion"]
        else:
            components = []

        # Filter and reorder according to component_order
        ordered_components = [c for c in self.component_order if c in components]
        return ordered_components

    def get_weights_for_task(self, task_type):
        """Get normalized weights for a specific task type."""
        components = self.get_components_for_task(task_type)

        # Extract weights for components or use equal weighting
        if not self.weights:
            # Equal weights (1/n for each component)
            task_weights = {comp: 1.0 / len(components) for comp in components}
        else:
            # Use provided weights, normalize to sum to 1
            task_weights = {comp: self.weights.get(comp, 0.25) for comp in components}
            weight_sum = sum(task_weights.values())
            task_weights = {comp: weight / weight_sum for comp, weight in task_weights.items()}

        return task_weights

    """# ===========================================================================
    # Model output (evaluator input) pre-processing
    # ==========================================================================="""

    def format_user_content(self, sample_row: dict) -> dict:
        """
        Takes a Huggingface Dataset row (dict), returns the formatted chat template user content string.
        Args:
            sample_row (dict): A row in Huggingface dataset that contains 'input', 'output', 'task_type' columns
        Returns:
            dict of strings containing extracted dataset fields and the formatted chat template user content string
        """
        input_text = sample_row['input']
        ground_truth = sample_row['output']
        task_type = sample_row['task_type']
        prompt = self.prompt_dict[task_type]
        user_content = sample_row['text']

        return {'input_text': input_text,
                'ground_truth': ground_truth,
                'task_type': task_type,
                'prompt': prompt,
                'user_content': user_content}

    def process_chunks_in_list(self, list_of_strings):
        output = []
        for item in list_of_strings:
            item = item.strip()
            if isinstance(item, str) and item[0] == '[' and item[-1] == ']':
                content = item[1:-1].strip().replace('\n', '')
                output.append([i.strip() + (">" if i[-1] != '>' else '') for i in content.split('>,')])
        return output

    def process_pred_str(self, pred_str: str, handle_format_error: bool = False) -> list:
        """
        Process a single generated output string representing a list of extracted ABSA elements (wrapped in <...></..> tags)
        Args:
            pred_str (str): model prediction output string representing a list.  E.g. "[<opn>Fun</opn>, <opn>interesting\'s topics</opn>]"
            handle_format_error (bool): if True, will handle (to some extent) variations in raw output such as missing closing tag, unexpected/missing quotation marks
        Returns:
            a list of strings, one string per each extracted chunk.  E.g. ['<opn>Fun</opn>', "<opn>interesting's topics</opn>"]
        """
        # basic cleaning
        pred_str = pred_str.replace('```', '')  # Remove all backticks
        pred_str = pred_str.replace('\n', ' ')  # Replace newlines with spaces
        pred_str = pred_str.replace('\t', ' ')  # Replace tabs with spaces
        pred_str = ' '.join(pred_str.split())  # Normalize all whitespace
        pred_str = pred_str.strip()

        # --------------------------------------------
        # No handling of output format error
        # --------------------------------------------
        if not handle_format_error:
            first_bracket = pred_str.find('[')
            last_bracket = pred_str.rfind(']')

            if first_bracket != -1 and last_bracket != -1 and first_bracket < last_bracket:
                pred_str = pred_str[first_bracket:last_bracket + 1]

            # Original logic
            if isinstance(pred_str, str) and pred_str[0] == '[' and pred_str[-1] == ']':
                content = pred_str[1:-1].strip()
                try:
                    # return [i.strip() + (">" if i[-1] != '>' else '') for i in content.split('>,')]
                    return [i.strip() for i in content.split(',')]
                except Exception as e:
                    print(f"Error processing: {pred_str}")
                    print(f"Exception: {e}")
                    return [pred_str]
            else:
                return [pred_str]

        else:
            # --------------------------------------------
            # Graceful handling of output format error
            # --------------------------------------------
            # Remove outer quotes if present
            if (pred_str.startswith("'") and pred_str.endswith("'")) or \
                    (pred_str.startswith('"') and pred_str.endswith('"')):
                pred_str = pred_str[1:-1].strip()

            # Remove curly braces if present
            if pred_str.startswith('{') and pred_str.endswith('}'):
                pred_str = pred_str[1:-1].strip()

            # Remove any additional outer quotes
            if (pred_str.startswith("'") and pred_str.endswith("'")) or \
                    (pred_str.startswith('"') and pred_str.endswith('"')):
                pred_str = pred_str[1:-1].strip()

            # Only process if it looks like a list (starts with '[')
            if not (isinstance(pred_str, str) and pred_str.startswith('[')):
                return [pred_str]

            # Find the content between brackets (handle truncated cases)
            bracket_end = pred_str.rfind(']')
            if bracket_end == -1:
                bracket_end = len(pred_str)
            else:
                bracket_end += 1

            content = pred_str[1:bracket_end - 1].strip()

            if not content:
                return []

            try:
                # Split by comma, but only when not inside angle brackets
                items = []
                current_item = ""
                inside_angle_brackets = False

                for char in content:
                    if char == '<':
                        inside_angle_brackets = True
                        current_item += char
                    elif char == '>':
                        inside_angle_brackets = False
                        current_item += char
                    elif char == ',' and not inside_angle_brackets:
                        # This comma is safe to split on - it's outside tags
                        if current_item.strip():
                            items.append(current_item.strip())
                        current_item = ""
                    else:
                        current_item += char

                # Add the last item
                if current_item.strip():
                    items.append(current_item.strip())

                # Clean up quotes around each item
                cleaned_items = []
                for item in items:
                    item = item.strip()
                    # Remove wrapping quotes
                    if (item.startswith('"') and item.endswith('"')) or \
                            (item.startswith("'") and item.endswith("'")):
                        item = item[1:-1]
                    cleaned_items.append(item)

                return cleaned_items

            except Exception as e:
                print(f"Error processing: {pred_str}")
                print(f"Exception: {e}")
                return [pred_str]

    """#===========================================================================
    # utility_functions
    #==========================================================================="""

    def _tokenize(self, text, stopwords):
        """
        Tokenize text by splitting on whitespace and stripping punctuation.

        Args:
            text: Input text string
            stopwords: List of stopwords to filter out

        Returns:
            List of tokens with punctuation stripped and stopwords removed
        """
        tokens = []
        for word in text.lower().split():
            # Strip punctuation from both ends
            cleaned = word.strip(string.punctuation).strip()
            # Keep non-empty tokens that aren't stopwords
            if cleaned and cleaned not in stopwords:
                tokens.append(cleaned)
        return tokens

    def _normalize_empty(self, text: str):
        """Normalize None to empty string and otherwise to lower-case."""
        if not text:
            return ""

        # Remove '*' from the punctuation set (i.e. skip consecutive * when removing trailing punctuations, e.g. in "such an ****")
        punctuation_to_strip = string.punctuation.replace('*', '')

        # Remove trailing punctuations, and then turn to lower case.
        cleaned = text.rstrip(punctuation_to_strip).strip().lower()

        return cleaned

    def flexible_text_similarity(self, pred, gold, input_text):
        """Calculate Flexible Text Similarity (FTS) score between predicted and gold strings."""

        # Cast None to empty string, strip extra white space, turn to lower-case
        pred_lower = self._normalize_empty(pred)
        gold_lower = self._normalize_empty(gold)

        # 1) Both empty: full match
        if not pred_lower and not gold_lower:
            return 1.0
        # 2) Reject if only one empty
        elif not pred_lower or not gold_lower:
            return 0.0
        # 3) Reject if pred is not in implicit_aspect_tokens and not in input_text (hallucination check)
        elif pred_lower and (pred_lower not in input_text.lower()) and (pred_lower not in self.implicit_aspect_tokens):
            return 0.0
        # 4) Reject if no overlap at all
        elif not set(pred_lower.split()).intersection(set(gold_lower.split())):
            return 0.0

        # Filter stopwords to create token lists
        pred_tokens = [t for t in pred_lower.split() if t not in self.stopwords]
        gold_tokens = [t for t in gold_lower.split() if t not in self.stopwords]

        # 5) Token lists match after stopword filtering:  full match
        if pred_tokens == gold_tokens:
            return 1.0
        # 6) Reject if no overlap after stopword filtering
        elif not set(pred_tokens).intersection(set(gold_tokens)):
            return 0.0
        # 7) Last case: partial overlap, use RougeL fmeasure
        else:
            pred_filtered = " ".join(pred_tokens)
            gold_filtered = " ".join(gold_tokens)

            scores = self.scorer.score(gold_filtered, pred_filtered)
            return scores['rougeL'].fmeasure

    def _check_unit_match(self, component_similarities, components, component_thresholds):
        """
        Args:
            component_similarities: Dict of component similarity scores
            components: List of components for this task type
            component_thresholds: Dict mapping component -> threshold dict with 'high_match' key
        """
        # Check sentiment (if present) - always requires exact match
        if "sentiment" in components:
            if component_similarities.get("sentiment", 0.0) < 1.0:
                return False

        # Check category (if present) - requires exact match (or thresholded partial match)
        if "category" in components:
            cat_sim = component_similarities.get("category", 0.0)
            category_threshold = self.partial_category_score if self.allow_partial_category_for_unit_match else 1.0
            if cat_sim < category_threshold:
                return False

        # Check aspect/opinion with their specific thresholds
        for comp in ["aspect", "opinion"]:
            if comp in components:
                comp_threshold = component_thresholds.get(comp, {}).get("high_match", 0.75)
                if component_similarities.get(comp, 0.0) < comp_threshold:
                    return False

        return True

    def calculate_component_similarity(self, pred_comp: str, gold_comp: str, component_type: str,
            input_text: str) -> float:
        """Calculate similarity between two components."""
        if component_type == "sentiment":
            # Exact match for sentiment
            return 1.0 if pred_comp.lower() == gold_comp.lower() else 0.0
        elif component_type == "category":
            # Check if both have hyphens
            if self.category_delimiter in pred_comp and self.category_delimiter in gold_comp:
                main_pred, sub_pred = [i.strip().lower() for i in pred_comp.split(self.category_delimiter, 1)]
                main_gold, sub_gold = [i.strip().lower() for i in gold_comp.split(self.category_delimiter, 1)]

                # If main categories match but subcategories don't, give partial credit
                if main_pred == main_gold:
                    if sub_pred == sub_gold:
                        return 1.0  # Full match
                    else:
                        return self.partial_category_score  # Partial match for same main category
                return 0.0  # No match
            else:
                # Exact match for regular categories
                return 1.0 if pred_comp.lower() == gold_comp.lower() else 0.0
        else:
            # FTS score for aspect and opinion
            similarity = self.flexible_text_similarity(pred_comp, gold_comp, input_text)
            return similarity

    def _extract_tag(self, text, tag_name):
        """Extract content from tags."""
        pattern = f"<{tag_name}>(.*?)</{tag_name}>"
        match = re.search(pattern, text)
        return match.group(1) if match else ""

    def extract_items(self, text_input):
        """Extract individual items from text input."""
        if not text_input or text_input.strip() == "":
            return []

        # Handle potential list string format
        if text_input.startswith("[") and text_input.endswith("]"):
            try:
                # Try to parse as a Python list string
                items = ast.literal_eval(text_input)
            except (SyntaxError, ValueError):
                # Fallback to regex-based extraction
                pattern = r"'([^']*)'|\"([^\"]*)\""
                items = [match.group(1) or match.group(2) for match in re.finditer(pattern, text_input)]
        else:
            # Assume comma or newline separated items
            items = [item.strip() for item in re.split(r',|\n', text_input) if item.strip()]

        return items

    def detect_task_type(self, text):
        """Detect the task type based on the tags in the text."""
        has_aspect = "<asp>" in text
        has_opinion = "<opn>" in text
        has_category = "<cat>" in text
        has_sentiment = "<sen>" in text or "<sent>" in text

        if has_aspect and has_opinion and has_category and has_sentiment:
            return "ASQE"
        elif has_aspect and has_opinion and has_sentiment:
            return "ASTE"
        elif has_aspect and has_opinion and has_category:
            return "AOC"
        elif has_aspect and has_opinion:
            return "AOPE"
        elif has_opinion:
            return "OE"
        else:
            return "unknown"

    def parse_items(self, input_data, task_type=None):
        """Parse items from input data (string or list)."""
        # Handle string or list input
        if isinstance(input_data, str):
            items = self.extract_items(input_data)
        else:
            items = input_data

        # Determine task type if not provided
        if task_type is None:
            if items:
                task_type = self.detect_task_type(items[0] if isinstance(items, list) else str(items))
            else:
                task_type = "unknown"

        parsed_items = []
        for item in items:
            parsed_item = {}

            # Extract components based on task type
            if task_type in ["ASQE", "AOC", "ASTE", "AOPE"]:
                aspect = self._extract_tag(item, "asp")
                if aspect:
                    parsed_item["aspect"] = aspect
                elif task_type != "OE":  # Skip items without aspect for non-OE tasks
                    continue

            # All task types have opinion
            opinion = self._extract_tag(item, "opn")
            if not opinion:
                continue
            parsed_item["opinion"] = opinion

            if task_type in ["ASQE", "AOC"]:
                category = self._extract_tag(item, "cat")
                if not category:
                    continue
                parsed_item["category"] = category

            if task_type in ["ASTE", "ASQE"]:
                sentiment = self._extract_tag(item, "sen") or self._extract_tag(item, "sent")
                if not sentiment:
                    continue
                parsed_item["sentiment"] = sentiment

            parsed_items.append(parsed_item)

        return parsed_items, task_type

    def get_match_quality(self, score):
        """Helper function to determine match quality."""
        if score >= 0.95:
            return "Full Match"
        elif score >= 0.8:
            return "Strong Match"
        elif score >= 0.6:
            return "Good Match"
        elif score >= 0.4:
            return "Partial Match"
        elif score > 0:
            return "Weak Match"
        else:
            return "No Match"

    def _create_empty_metrics(self, entry_id, task_type, n_pred, n_gold):
        """Create metrics for empty predictions or gold items."""
        components = self.get_components_for_task(task_type)

        metric = {
            "entry_id": entry_id,
            "task_type": task_type,
            "n_pairs": 0,
            "pred_count": n_pred,
            "gold_count": n_gold,
            "matched_count": 0,

            "weighted_component_precision": 0.0,
            "weighted_component_recall": 0.0,
            "weighted_component_f1": 0.0
        }

        # Add component metrics (all zeros)
        for component in components:
            # Add count fields that _calculate_task_summary expects
            metric[f"{component}_tp"] = 0
            metric[f"{component}_fp"] = 0
            metric[f"{component}_fn"] = 0
            metric[f"{component}_precision"] = 0.0
            metric[f"{component}_recall"] = 0.0
            metric[f"{component}_f1"] = 0.0

        # -------unit_match_update-------
        # Add unit_match metrics - handle empty cases correctly
        if n_pred == 0 and n_gold == 0:
            # Both empty - no errors
            metric["unit_match_tp"] = 0
            metric["unit_match_fp"] = 0
            metric["unit_match_fn"] = 0
        elif n_pred == 0:
            # No predictions, have gold items - all FN
            metric["unit_match_tp"] = 0
            metric["unit_match_fp"] = 0
            metric["unit_match_fn"] = n_gold
        elif n_gold == 0:
            # Have predictions, no gold items - all FP
            metric["unit_match_tp"] = 0
            metric["unit_match_fp"] = n_pred
            metric["unit_match_fn"] = 0
        else:
            # Both non-empty but no matches (shouldn't happen normally)
            metric["unit_match_tp"] = 0
            metric["unit_match_fp"] = n_pred
            metric["unit_match_fn"] = n_gold

        metric["unit_match_precision"] = 0.0
        metric["unit_match_recall"] = 0.0
        metric["unit_match_f1"] = 0.0
        # -------unit_match_update-------

        return metric

    def _reorder_columns(self, df):
        """Reorder DataFrame columns to put components in the specified order."""
        if df.empty:
            return df

        # Define column groups with new ordering
        id_cols = ["entry_id", "task_type", "pair_id", "n_pairs"]
        text_cols = ["input_text", "gold", "pred"]
        count_cols = [col for col in df.columns if "count" in col or "total" in col]

        # Early metrics: unit_match, weighted_f1, and weighted_macro metrics
        early_metric_cols = []

        # Add unit_match columns if they exist - group micro then macro
        unit_match_cols = [col for col in df.columns if "unit_match" in col]

        if unit_match_cols:
            unit_match_ordered = []
            # Add TP, FP, FN first
            for suffix in ["_tp", "_fp", "_fn", "_TP", "_FP", "_FN"]:
                unit_match_ordered.extend([col for col in unit_match_cols if col.endswith(suffix)])
            # Add micro metrics
            for suffix in ["_micro_precision", "_micro_recall", "_micro_f1"]:
                unit_match_ordered.extend([col for col in unit_match_cols if col.endswith(suffix)])
            # Add macro metrics
            for suffix in ["_macro_precision", "_macro_recall", "_macro_f1"]:
                unit_match_ordered.extend([col for col in unit_match_cols if col.endswith(suffix)])
            # Add any remaining unit_match columns
            unit_match_ordered.extend([col for col in unit_match_cols if col not in unit_match_ordered])
            early_metric_cols.extend(unit_match_ordered)

        # Add weighted_component metrics right after unit_match
        # For entries_metrics_df: weighted_component_precision, _recall, _f1
        for metric in ["weighted_component_precision", "weighted_component_recall", "weighted_component_f1"]:
            if metric in df.columns:
                early_metric_cols.append(metric)

        # For task_summary_df: weighted_component_macro_precision, _recall, _f1
        weighted_component_macro_cols = [col for col in df.columns if col.startswith("weighted_component_macro")]
        if weighted_component_macro_cols:
            # Order: precision, recall, f1
            for metric in ["weighted_component_macro_precision", "weighted_component_macro_recall",
                           "weighted_component_macro_f1"]:
                if metric in weighted_component_macro_cols:
                    early_metric_cols.append(metric)

        # Group component columns (aspect, opinion, category, sentiment)
        component_col_groups = []
        for component in self.component_order:
            component_cols = [col for col in df.columns
                              if component in col
                              and "unit_match" not in col
                              and "weighted_component_macro" not in col
                              and "weighted_component" not in col]  # Exclude all weighted_component columns
            if component_cols:
                component_col_groups.append(component_cols)

        # Flatten component columns while preserving order
        ordered_component_cols = [col for group in component_col_groups for col in group]

        # Get remaining columns
        other_cols = [col for col in df.columns
                      if col not in id_cols + text_cols + count_cols + early_metric_cols + ordered_component_cols]

        # Create final column order: id, text, counts, early_metrics (unit_match + weighted_f1 + weighted_macro), components, other
        ordered_cols = id_cols + text_cols + count_cols + early_metric_cols + ordered_component_cols + other_cols

        # Only keep columns that exist in the DataFrame
        final_cols = [col for col in ordered_cols if col in df.columns]

        return df[final_cols]

    def _find_best_match_pair_id(self, entry_id, matched_pairs, similarity_matrix, pred_items, gold_items):
        """
        Find the best match pair ID for entry-level metrics.

        Args:
            entry_id: The ID of the current entry
            matched_pairs: List of (pred_idx, gold_idx) tuples for matches
            similarity_matrix: Matrix of similarity scores
            pred_items: List of predicted items
            gold_items: List of gold items

        Returns:
            best_match_pair_id: Formatted pair ID for the best match
        """
        if matched_pairs:
            best_match_idx = 0
            best_match_score = -1

            # Find the pair with the highest similarity score
            for i, (p_idx, g_idx) in enumerate(matched_pairs):
                score = similarity_matrix[p_idx, g_idx]
                if score > best_match_score:
                    best_match_score = score
                    best_match_idx = i

            p_idx, g_idx = matched_pairs[best_match_idx]
            best_match_pair_id = f"{entry_id}-g{g_idx}-p{p_idx}"
        else:
            # Handle cases with no matches
            if len(gold_items) > 0:
                best_match_pair_id = f"{entry_id}-g0-p_"  # Has gold items but no predictions matched
            elif len(pred_items) > 0:
                best_match_pair_id = f"{entry_id}-g_-p0"  # Has predictions but no gold items matched
            else:
                best_match_pair_id = f"{entry_id}-g_-p_"  # Neither gold nor predictions

        return best_match_pair_id

    """# ===========================================================================
    # Core Algorithm Functions
    # ==========================================================================="""

    # Helper function of bipartite_matching
    def calculate_pair_component_similarities(self, pred_item, gold_item, task_type, input_text):
        """Calculate similarities for all components and return both individual similarities and weighted similarity."""

        # pred_item, gold_item: e.g. {'aspect': '...', 'category': '...', 'opinion': '...'}
        components = self.get_components_for_task(task_type)
        weights = self.get_weights_for_task(task_type)

        # Calculate component similarities
        component_similarity_dict = {}
        for component in components:
            similarity = self.calculate_component_similarity(pred_item[component], gold_item[component], component,
                                                             input_text)
            component_similarity_dict[component] = similarity

        # Calculate weighted similarity
        weighted_similarity_score = sum(
            weights[component] * component_similarity_dict[component] for component in components)

        return component_similarity_dict, weighted_similarity_score

    def bipartite_matching(self, pred_items, gold_items, task_type, input_text) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, dict]:
        """
        Perform minimum-weight bipartite matching between predicted and gold items, identify optimal matches based on component-wise similarity.
        Args:
            pred_items (list[dict]): list of dicts, each dict is a model-predicted opinion/pair/triplet/quadruplet
            gold_items (list[dict]): list of dicts, each dict is a 'ground-truth' opinion/pair/triplet/quadruplet
            task_type (str): one of 'OE', 'AOPE', 'AOC', 'ASTE', 'ASQE', 'SC'.
            input_text (str): the original input text for hallucination checking

        Returns: similarity_matrix, row_ind, col_ind, component_similarities_dict
        """
        n_pred = len(pred_items)
        n_gold = len(gold_items)

        if n_pred == 0 or n_gold == 0:
            return np.zeros((max(1, n_pred), max(1, n_gold))), np.array([]), np.array([]), {}

        # Calculate similarity for all pairs
        similarity_matrix = np.zeros((n_pred, n_gold))
        component_similarities_dict = {}  # Store component similarities for reuse

        for i, pred in enumerate(pred_items):
            for j, gold in enumerate(gold_items):
                pair_component_similarity_dict, pair_weighted_similarity_score = self.calculate_pair_component_similarities(
                    pred, gold, task_type, input_text
                )
                similarity_matrix[i, j] = pair_weighted_similarity_score
                component_similarities_dict[(i, j)] = pair_component_similarity_dict

        # Convert to cost matrix (Hungarian algorithm minimizes cost)
        cost_matrix = 1 - similarity_matrix

        # Apply Hungarian algorithm / minimum-weight matching
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        return similarity_matrix, row_ind, col_ind, component_similarities_dict

    def calculate_component_stats(self, pred_items, gold_items, matched_pairs, task_type,
            component_similarities_dict=None, input_text=None):
        """
        Calculate detailed statistics for individual components within the same entry_id.

        Args:
            pred_items: List of predicted items (dictionaries)
            gold_items: List of gold/reference items (dictionaries)
            matched_pairs: List of tuples (pred_idx, gold_idx) for matched items
            task_type: Type of task ('OE', 'AOPE', 'AOC', 'ASTE', 'ASQE', 'SC')
            component_similarities_dict: Pre-computed component similarities
            input_text: Original input text for hallucination checking

        Returns:
            Dictionary containing component-level statistics
        """

        # Validate inputs
        if not isinstance(pred_items, list) or not isinstance(gold_items, list):
            raise TypeError("pred_items and gold_items must be lists")

        if not isinstance(matched_pairs, list):
            raise TypeError("matched_pairs must be a list of tuples")

        # Get relevant components for this task type
        components = self.get_components_for_task(task_type)
        if not components:
            raise ValueError(f"Invalid task type: {task_type}")

        # Initialize component statistics
        component_stats = {}
        for comp in components:
            component_stats[comp] = {
                "tp": 0, "fp": 0, "fn": 0,
                "matches": [],
                "scores": []
            }

        # -------unit_match_update-------
        # Initialize unit_match statistics
        component_stats["unit_match"] = {
            "tp": 0, "fp": 0, "fn": 0,
            "matches": [],
            "scores": []
        }
        # -------unit_match_update-------

        # Track unit match status for each matched pair
        unit_match_status = {}  # (p_idx, g_idx) -> bool

        # Process matched pairs with validation
        for p_idx, g_idx in matched_pairs:
            # Validate indices
            if p_idx >= len(pred_items) or g_idx >= len(gold_items):
                continue

            pred = pred_items[p_idx]
            gold = gold_items[g_idx]

            # Build component-specific thresholds dict for THIS pair
            component_threshold_dict = {}

            for component in components:
                # Skip if component is missing in either item (not incrementing component-metric count)
                if component not in pred or component not in gold:
                    continue

                # Use precomputed similarities when available to avoid redundant calculations
                if component_similarities_dict and (p_idx, g_idx) in component_similarities_dict:
                    similarity = component_similarities_dict[(p_idx, g_idx)][component]
                else:
                    # Need input_text to calculate similarity
                    if input_text is None:
                        raise ValueError("input_text is required when component_similarities_dict is not provided")
                    similarity = self.calculate_component_similarity(
                        pred[component], gold[component], component, input_text
                    )

                # Only get thresholds for aspect/opinion (category/sentiment use exact matching)
                if component in ["aspect", "opinion"]:
                    gold_tokens = [t for t in gold[component].strip().lower().split() if t not in self.stopwords]
                    gold_length = len(gold_tokens) if gold_tokens else 1
                    component_threshold_subdict = self.get_thresholds(gold_length)
                    component_threshold_dict[component] = component_threshold_subdict
                else:
                    # For category/sentiment, thresholds aren't used but store default for safety
                    component_threshold_dict[component] = self.get_thresholds()

                component_stats[component]["scores"].append(similarity)

                # Count as TP if meets high threshold
                # For category/sentiment, this is simply checking if similarity == 1.0
                if component in ["aspect", "opinion"]:
                    comp_threshold = component_threshold_dict[component]["high_match"]
                elif component == 'category' and self.allow_partial_category_for_unit_match:
                    comp_threshold = self.partial_category_score
                else:
                    comp_threshold = 1.0  # Exact match required for category/sentiment

                if similarity >= comp_threshold:
                    component_stats[component]["tp"] += 1
                    component_stats[component]["matches"].append((p_idx, g_idx))
                else:
                    component_stats[component]["fp"] += 1
                    component_stats[component]["fn"] += 1

            # After component loop - check unit match
            if component_similarities_dict and (p_idx, g_idx) in component_similarities_dict:
                pair_component_sims = component_similarities_dict[(p_idx, g_idx)]
            else:
                # Reconstruct if needed
                pair_component_sims = {}
                for comp in components:
                    if comp in pred and comp in gold:
                        pair_component_sims[comp] = self.calculate_component_similarity(
                            pred[comp], gold[comp], comp, input_text
                        )

            # Check if unit matches with component-specific thresholds
            is_unit_match = self._check_unit_match(
                pair_component_sims, components, component_threshold_dict
            )
            # Store unit match status
            unit_match_status[(p_idx, g_idx)] = is_unit_match

            # Update unit_match stats based on strict matching logic
            if is_unit_match:
                component_stats["unit_match"]["tp"] += 1
                component_stats["unit_match"]["matches"].append((p_idx, g_idx))
                component_stats["unit_match"]["scores"].append(1.0)
            else:
                # Unit doesn't match - counts as both FP and FN
                component_stats["unit_match"]["fp"] += 1
                component_stats["unit_match"]["fn"] += 1
                component_stats["unit_match"]["scores"].append(0.0)

        # Process unmatched items
        self._process_unmatched_items(pred_items, gold_items, matched_pairs, component_stats, components)

        # Calculate metrics
        for component, stats in component_stats.items():
            tp = stats["tp"]
            fp = stats["fp"]
            fn = stats["fn"]

            # Get precision, recall, and F1 metrics using helper method
            stats.update(self._calculate_precision_recall_f1(tp, fp, fn))

        # Store unit match status in component_stats for later use
        component_stats["_unit_match_status"] = unit_match_status

        return component_stats

    def _process_unmatched_items(self, pred_items, gold_items, matched_pairs, component_stats, components):
        """
        Process unmatched items to update FP and FN counts.

        Args:
            pred_items: List of predicted items
            gold_items: List of gold items
            matched_pairs: List of (pred_idx, gold_idx) tuples
            component_stats: Dictionary to update with unmatched statistics
            components: List of components to process
        """

        # Convert matched indices to sets for efficient lookup
        matched_pred_indices = set(p for p, _ in matched_pairs if p < len(pred_items))
        matched_gold_indices = set(g for _, g in matched_pairs if g < len(gold_items))

        # Add unmatched predictions as FP (with improved validation)
        for p_idx, pred_item in enumerate(pred_items):
            if p_idx not in matched_pred_indices:
                for component in components:
                    if component in pred_item and pred_item[component].strip():  # Only count non-empty components
                        component_stats[component]["fp"] += 1

                # -------unit_match_update-------
                # Unmatched prediction means the entire unit is FP
                component_stats["unit_match"]["fp"] += 1
                # -------unit_match_update-------

        # Add unmatched gold items as FN (with improved validation)
        for g_idx, gold_item in enumerate(gold_items):
            if g_idx not in matched_gold_indices:
                for component in components:
                    if component in gold_item and gold_item[component].strip():  # Only count non-empty components
                        component_stats[component]["fn"] += 1

                # -------unit_match_update-------
                # Unmatched gold item means the entire unit is FN
                component_stats["unit_match"]["fn"] += 1
                # -------unit_match_update-------

    # New helper function for precision, recall, F1 calculation
    def _calculate_precision_recall_f1(self, tp, fp, fn):
        """Calculate precision, recall, and F1 score from true positives, false positives, and false negatives."""
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    # New helper function to create pair metrics
    def _create_pair_metrics(self, entry_id, task_type, matched_pairs, component_similarities_dict,
            similarity_matrix, pred, gold, pairs_metrics, unit_match_status=None):
        """Create metrics for each matched pair. Updates pairs_metrics"""
        for p_idx, g_idx in matched_pairs:
            pair_component_sims_dict = component_similarities_dict[(p_idx, g_idx)]
            pair_weighted_sim_score = similarity_matrix[p_idx, g_idx]

            # Get unit match status for this pair
            is_unit_match = unit_match_status.get((p_idx, g_idx), False) if unit_match_status else False

            # Store only the specific matched pred and gold items instead of entire lists
            pair_id = f"{entry_id}-g{g_idx}-p{p_idx}"
            pair_metric = {
                "entry_id": entry_id,
                "task_type": task_type,
                "pair_id": pair_id,
                "gold_item": gold[g_idx] if g_idx < len(gold) else None,
                "pred_item": pred[p_idx] if p_idx < len(pred) else None,
                "weighted_score": pair_weighted_sim_score,
                "is_unit_match": is_unit_match,
                **{f'{component}_score': score for component, score in pair_component_sims_dict.items()}
            }

            pairs_metrics.append(pair_metric)

    def _create_entry_metrics(self, entry_id, task_type, matched_pairs, pred_items, gold_items, component_stats):
        """Create metrics for an entry using component statistics."""
        entry_metric = {
            "entry_id": entry_id,
            "task_type": task_type,
            "n_pairs": len(matched_pairs),
            "pred_count": len(pred_items),
            "gold_count": len(gold_items),
            "matched_count": len(matched_pairs)
        }

        # Get task-specific components (excludes unit_match)
        components = self.get_components_for_task(task_type)

        # Add component metrics
        for component in components:
            # -------unit_match_update-------
            # Add validation to prevent KeyError
            if component not in component_stats:
                # Defensive: component missing from stats (shouldn't happen in normal flow)
                entry_metric[f"{component}_tp"] = 0
                entry_metric[f"{component}_fp"] = 0
                entry_metric[f"{component}_fn"] = 0
                entry_metric[f"{component}_precision"] = 0.0
                entry_metric[f"{component}_recall"] = 0.0
                entry_metric[f"{component}_f1"] = 0.0
            else:
                entry_metric[f"{component}_tp"] = component_stats[component]["tp"]
                entry_metric[f"{component}_fp"] = component_stats[component]["fp"]
                entry_metric[f"{component}_fn"] = component_stats[component]["fn"]
                entry_metric[f"{component}_precision"] = component_stats[component]["precision"]
                entry_metric[f"{component}_recall"] = component_stats[component]["recall"]
                entry_metric[f"{component}_f1"] = component_stats[component]["f1"]
            # -------unit_match_update-------

            # Add similarity statistics
            scores = component_stats[component]["scores"]
            if scores:
                entry_metric[f"{component}_avg_similarity"] = np.mean(scores)
                entry_metric[f"{component}_std_similarity"] = np.std(scores)
            else:
                entry_metric[f"{component}_avg_similarity"] = 0.0
                entry_metric[f"{component}_std_similarity"] = 0.0

        # -------unit_match_update-------
        # Add unit_match metrics with validation
        if "unit_match" in component_stats:
            entry_metric["unit_match_tp"] = component_stats["unit_match"]["tp"]
            entry_metric["unit_match_fp"] = component_stats["unit_match"]["fp"]
            entry_metric["unit_match_fn"] = component_stats["unit_match"]["fn"]
            entry_metric["unit_match_precision"] = component_stats["unit_match"]["precision"]
            entry_metric["unit_match_recall"] = component_stats["unit_match"]["recall"]
            entry_metric["unit_match_f1"] = component_stats["unit_match"]["f1"]
        else:
            # Defensive: unit_match missing (shouldn't happen with updated calculate_component_stats)
            entry_metric["unit_match_tp"] = 0
            entry_metric["unit_match_fp"] = 0
            entry_metric["unit_match_fn"] = 0
            entry_metric["unit_match_precision"] = 0.0
            entry_metric["unit_match_recall"] = 0.0
            entry_metric["unit_match_f1"] = 0.0
        # -------unit_match_update-------

        # Calculate weighted component F1 score (ONLY from task components, NOT unit_match)
        weights = self.get_weights_for_task(task_type)

        weighted_component_precision = sum(weights[comp] * component_stats[comp]["precision"]
                                           for comp in components if comp in component_stats)
        weighted_component_recall = sum(weights[comp] * component_stats[comp]["recall"]
                                        for comp in components if comp in component_stats)
        weighted_component_f1 = sum(weights[comp] * component_stats[comp]["f1"]
                                    for comp in components if comp in component_stats)

        entry_metric["weighted_component_precision"] = weighted_component_precision
        entry_metric["weighted_component_recall"] = weighted_component_recall
        entry_metric["weighted_component_f1"] = weighted_component_f1

        return entry_metric

    # New helper function to calculate task summary
    def _calculate_task_summary(self, task_summary_data):
        """Calculate aggregated task-level metrics."""
        task_summary = []

        for task_type, metrics in task_summary_data.items():
            components = self.get_components_for_task(task_type)

            # Calculate aggregated counts
            tp_sum = defaultdict(int)
            fp_sum = defaultdict(int)
            fn_sum = defaultdict(int)

            for metric in metrics:
                for component in components:
                    tp_sum[component] += metric[f"{component}_tp"]
                    fp_sum[component] += metric[f"{component}_fp"]
                    fn_sum[component] += metric[f"{component}_fn"]

                # -------unit_match_update-------
                # Aggregate unit_match counts
                tp_sum["unit_match"] += metric["unit_match_tp"]
                fp_sum["unit_match"] += metric["unit_match_fp"]
                fn_sum["unit_match"] += metric["unit_match_fn"]
                # -------unit_match_update-------

            # Create task metric
            task_metric = {
                "task_type": task_type,
                "entry_count": len(metrics),
                "total_pred": sum(m["pred_count"] for m in metrics),
                "total_gold": sum(m["gold_count"] for m in metrics),
                "total_matched": sum(m["matched_count"] for m in metrics),
            }

            # Calculate micro and macro metrics
            for component in components:
                self._add_component_metrics(task_metric, component, tp_sum, fp_sum, fn_sum, metrics)

                # Add similarity statistics across all entries
                all_scores = []
                for metric in metrics:
                    score_key = f"{component}_avg_similarity"
                    if score_key in metric and metric[score_key] > 0:  # Only include entries with matches
                        all_scores.append(metric[score_key])

                if all_scores:
                    task_metric[f"{component}_overall_avg_similarity"] = np.mean(all_scores)
                    task_metric[f"{component}_overall_std_similarity"] = np.std(all_scores)
                else:
                    task_metric[f"{component}_overall_avg_similarity"] = 0.0
                    task_metric[f"{component}_overall_std_similarity"] = 0.0

            # -------unit_match_update-------
            # Calculate micro and macro metrics for unit_match
            self._add_component_metrics(task_metric, "unit_match", tp_sum, fp_sum, fn_sum, metrics)
            # -------unit_match_update-------

            # Calculate weighted component macro metrics (not including unit_match)
            weights = self.get_weights_for_task(task_type)

            if metrics:
                weighted_component_macro_precision = sum(
                    weights[comp] * (sum(m[f"{comp}_precision"] for m in metrics) / len(metrics))
                    for comp in components
                )

                weighted_component_macro_recall = sum(
                    weights[comp] * (sum(m[f"{comp}_recall"] for m in metrics) / len(metrics))
                    for comp in components
                )

                weighted_component_macro_f1 = sum(m["weighted_component_f1"] for m in metrics) / len(metrics)
            else:
                weighted_component_macro_precision = 0
                weighted_component_macro_recall = 0
                weighted_component_macro_f1 = 0

            task_metric["weighted_component_macro_precision"] = weighted_component_macro_precision
            task_metric["weighted_component_macro_recall"] = weighted_component_macro_recall
            task_metric["weighted_component_macro_f1"] = weighted_component_macro_f1

            task_summary.append(task_metric)

        return task_summary

    # New helper function to add component metrics to task summary
    def _add_component_metrics(self, task_metric, component, tp_sum, fp_sum, fn_sum, metrics):
        """Add micro and macro metrics for a component to the task summary."""
        # Micro metrics
        micro_metrics = self._calculate_precision_recall_f1(
            tp_sum[component], fp_sum[component], fn_sum[component]
        )

        task_metric[f"{component}_TP"] = tp_sum[component]
        task_metric[f"{component}_FP"] = fp_sum[component]
        task_metric[f"{component}_FN"] = fn_sum[component]

        task_metric[f"{component}_micro_precision"] = micro_metrics["precision"]
        task_metric[f"{component}_micro_recall"] = micro_metrics["recall"]
        task_metric[f"{component}_micro_f1"] = micro_metrics["f1"]

        # Macro metrics
        if metrics:
            task_metric[f"{component}_macro_precision"] = sum(m[f"{component}_precision"] for m in metrics) / len(
                metrics)
            task_metric[f"{component}_macro_recall"] = sum(m[f"{component}_recall"] for m in metrics) / len(metrics)
            task_metric[f"{component}_macro_f1"] = sum(m[f"{component}_f1"] for m in metrics) / len(metrics)
        else:
            task_metric[f"{component}_macro_precision"] = 0
            task_metric[f"{component}_macro_recall"] = 0
            task_metric[f"{component}_macro_f1"] = 0

    """# ===========================================================================
    # Output and Visualisation
    # ==========================================================================="""

    def generate_match_details(self, predictions, ground_truth, task_type=None,
            parsed_items=None, similarity_matrix=None, bipartite_indices=None,
            component_similarities_dict=None, input_text=None):
        """
        Generate details of matched pairs between predictions and ground truth.

        Args:
            predictions: List of prediction strings or parsed items
            ground_truth: List of ground truth strings or parsed items
            task_type: Type of task ('OE', 'AOPE', 'AOC', 'ASTE', 'ASQE', 'SC')
            parsed_items: Tuple of (pred_items, gold_items) if already parsed
            similarity_matrix: Pre-computed similarity matrix
            bipartite_indices: Tuple of (row_ind, col_ind) from linear_sum_assignment
            component_similarities_dict: Pre-computed component similarities
            input_text: Original input text for hallucination checking

        Returns:
            Dictionary with match details
        """

        # Parse inputs if not provided
        if parsed_items is not None:
            pred_items, gold_items = parsed_items
        else:
            pred_items, pred_task_type = self.parse_items(predictions, task_type)
            gold_items, gold_task_type = self.parse_items(ground_truth, task_type)

            # Determine task type with validation
            if task_type is None:
                if pred_task_type != gold_task_type and pred_items and gold_items:
                    raise ValueError(
                        f"Task type mismatch: predictions are {pred_task_type}, ground truth is {gold_task_type}")
                task_type = gold_task_type if gold_task_type != "unknown" else pred_task_type

            if task_type == "unknown":
                raise ValueError("Could not determine task type from inputs")

        # If empty inputs, return empty result
        if not pred_items or not gold_items:
            return {"task_type": task_type, "matches": []}

        # Calculate similarity matrix if not provided
        if similarity_matrix is None or bipartite_indices is None or component_similarities_dict is None:
            if input_text is None:
                raise ValueError("input_text is required when similarity_matrix is not provided")
            else:
                similarity_matrix, row_ind, col_ind, component_similarities_dict = self.bipartite_matching(
                    pred_items, gold_items, task_type, input_text
                )

        else:
            row_ind, col_ind = bipartite_indices

        # # Get components for this task type
        # components = self.get_components_for_task(task_type)

        # Create match information using helper function
        matches = self._create_match_information(
            pred_items, gold_items, row_ind, col_ind,
            similarity_matrix, component_similarities_dict
        )

        result = {
            "task_type": task_type,
            "matches": matches,
            "gold_items": gold_items,
            "pred_items": pred_items,
            "bipartite_indices": (row_ind, col_ind)
        }

        return result

    def _create_match_information(self, pred_items, gold_items, row_ind, col_ind,
            similarity_matrix, component_similarities_dict, unit_match_status=None):
        """Create detailed match information for optimal and non-optimal matches."""
        matches = []

        # Add optimal matches from Hungarian algorithm
        for i in range(len(row_ind)):
            p_idx = row_ind[i]
            g_idx = col_ind[i]

            # Ensure indices are within bounds
            if p_idx >= len(pred_items) or g_idx >= len(gold_items):
                continue

            # Create match info for optimal match
            pair_id = f"g{g_idx}-p{p_idx}"
            overall_score = float(similarity_matrix[p_idx, g_idx])
            component_scores = component_similarities_dict.get((p_idx, g_idx), {})

            # Get unit match status for this pair
            is_unit_match = unit_match_status.get((p_idx, g_idx), False) if unit_match_status else False

            match_info = {
                "pred_index": int(p_idx),
                "gold_index": int(g_idx),
                "pair_id": pair_id,
                "overall_score": overall_score,
                "component_scores": component_scores,
                "pred_item": pred_items[p_idx],
                "gold_item": gold_items[g_idx],
                "is_optimal_match": True,
                "is_full_match": overall_score >= self.thresholds["overall_weighted"],
                "is_unit_match": is_unit_match
            }
            matches.append(match_info)

        # Add non-optimal matches above threshold
        self._add_non_optimal_matches(
            matches, pred_items, gold_items, row_ind, col_ind,
            similarity_matrix, component_similarities_dict
        )

        # Sort matches by score (highest first)
        matches.sort(key=lambda x: x["overall_score"], reverse=True)

        return matches

    def _add_non_optimal_matches(self, matches, pred_items, gold_items, row_ind, col_ind,
            similarity_matrix, component_similarities_dict):
        """Add non-optimal matches that are above the threshold."""
        # Track optimal matches
        optimal_pairs = set((row_ind[i], col_ind[i]) for i in range(len(row_ind))
                            if row_ind[i] < len(pred_items) and col_ind[i] < len(gold_items))

        # Check all possible pred-gold pairs
        for p_idx in range(len(pred_items)):
            for g_idx in range(len(gold_items)):
                # Skip optimal matches
                if (p_idx, g_idx) in optimal_pairs:
                    continue

                if p_idx < similarity_matrix.shape[0] and g_idx < similarity_matrix.shape[1]:
                    score = float(similarity_matrix[p_idx, g_idx])
                    # Create non-optimal match info
                    pair_id = f"g{g_idx}-p{p_idx}-nonOpt"
                    component_scores = component_similarities_dict.get((p_idx, g_idx), {})

                    match_info = {
                        "pred_index": int(p_idx),
                        "gold_index": int(g_idx),
                        "pair_id": pair_id,
                        "overall_score": score,
                        "component_scores": component_scores,
                        "pred_item": pred_items[p_idx],
                        "gold_item": gold_items[g_idx],
                        "is_optimal_match": False,
                        "is_full_match": score >= self.thresholds["overall_weighted"]
                    }
                    matches.append(match_info)

    def create_match_df(self, match_details_dict):
        """Create a comprehensive DataFrame showing all matched and unmatched items with pair IDs."""

        data = []

        for entry_id, matches_dict in match_details_dict.items():
            task_type = matches_dict["task_type"]
            all_matches = matches_dict.get("matches", [])
            gold_items = matches_dict.get("gold_items", [])
            pred_items = matches_dict.get("pred_items", [])
            input_text = matches_dict.get("input_text", "")

            components = self.get_components_for_task(task_type)

            # Find all optimal matches
            optimal_matches = {}
            optimal_pred_indices = set()
            is_optimal_match = None

            # First, collect all matches that are marked as optimal
            for match in all_matches:

                if match["is_optimal_match"]:
                    gold_idx = match["gold_index"]
                    pred_idx = match["pred_index"]
                    optimal_matches[gold_idx] = {
                        "pred_idx": pred_idx,
                        "score": match["overall_score"],
                        "component_scores": match["component_scores"],
                        "pair_id": f"{entry_id}-g{gold_idx}-p{pred_idx}",
                        "is_unit_match": match.get("is_unit_match", False)
                    }
                    optimal_pred_indices.add(pred_idx)

            # 1. Create rows for all gold items (matched or unmatched)
            for gold_idx, gold_item in enumerate(gold_items):
                if gold_idx in optimal_matches:  # alraedy matched
                    # Gold item has a match
                    match_info = optimal_matches[gold_idx]
                    pred_idx = match_info["pred_idx"]
                    pred_item = pred_items[pred_idx]
                    score = match_info["score"]
                    component_scores = match_info["component_scores"]
                    pair_id = match_info["pair_id"]
                    is_unit_match = match_info["is_unit_match"]

                    # Create row for the pair
                    pair_row = {
                        "entry_id": entry_id,
                        "Task": task_type,
                        "PairID": pair_id,
                        "is_optimal_match": True,  # matched
                        "is_unit_match": is_unit_match,
                        "Input Text": input_text,
                        "Overall Score": score,
                        "Match Quality": self.get_match_quality(score)
                    }

                    # Add component values and scores (clustered together)
                    for comp in components:
                        if comp in gold_item:
                            pair_row[f"Gold-{comp.capitalize()}"] = gold_item[comp]
                        else:
                            pair_row[f"Gold-{comp.capitalize()}"] = "[N/A]"

                        if comp in pred_item:
                            pair_row[f"Pred-{comp.capitalize()}"] = pred_item[comp]
                        else:
                            pair_row[f"Pred-{comp.capitalize()}"] = "[N/A]"

                        # Add component score
                        comp_score = component_scores.get(comp, 0.0)
                        pair_row[f"{comp.capitalize()} Score"] = comp_score

                    data.append(pair_row)
                else:  # unmatched gold-items
                    # Gold item has no match - use underscore for missing pred index
                    pair_id = f"{entry_id}-g{gold_idx}-p_"

                    # Create row for the pair
                    pair_row = {
                        "entry_id": entry_id,
                        "Task": task_type,
                        "PairID": pair_id,
                        "is_optimal_match": False,  # unmatched gold
                        "is_unit_match": False,
                        "Input Text": input_text,
                        "Overall Score": 0.0,
                        "Match Quality": "No Match"
                    }

                    # Add component values
                    for comp in components:
                        if comp in gold_item:
                            pair_row[f"Gold-{comp.capitalize()}"] = gold_item[comp]
                        else:
                            pair_row[f"Gold-{comp.capitalize()}"] = "[N/A]"

                        pair_row[f"Pred-{comp.capitalize()}"] = "[N/A]"  # "null"

                        # Add component score (always zero)
                        pair_row[f"{comp.capitalize()} Score"] = 0.0

                    data.append(pair_row)

            # 2. Create rows for unmatched pred items
            for pred_idx, pred_item in enumerate(pred_items):
                if pred_idx not in optimal_pred_indices:  # unmatched pred_items
                    # Use underscore for missing gold index
                    pair_id = f"{entry_id}-g_-p{pred_idx}"

                    # Create row for the pair
                    pair_row = {
                        "entry_id": entry_id,
                        "Task": task_type,
                        "PairID": pair_id,
                        "is_optimal_match": False,  # unmatched pred
                        "is_unit_match": False,
                        "Input Text": input_text,
                        "Overall Score": 0.0,
                        "Match Quality": "No Match"
                    }

                    # Add component values
                    for comp in components:
                        pair_row[f"Gold-{comp.capitalize()}"] = "[N/A]"  # "null"

                        if comp in pred_item:
                            pair_row[f"Pred-{comp.capitalize()}"] = pred_item[comp]
                        else:
                            pair_row[f"Pred-{comp.capitalize()}"] = "[N/A]"

                        # Add component score (always zero)
                        pair_row[f"{comp.capitalize()} Score"] = 0.0

                    data.append(pair_row)

        # Create DataFrame
        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)

        # Arrange columns to group content and scores by component
        first_cols = ["entry_id", "Task", "PairID", "is_optimal_match",
                      "is_unit_match",
                      "Input Text", "Match Quality"]
        component_cols = []

        for comp in ["Aspect", "Opinion", "Category", "Sentiment"]:
            # Add clustered columns for each component (if they exist)
            gold_col = f"Gold-{comp}"
            pred_col = f"Pred-{comp}"
            score_col = f"{comp} Score"

            cols_to_add = []
            if gold_col in df.columns:
                cols_to_add.append(gold_col)
            if pred_col in df.columns:
                cols_to_add.append(pred_col)
            if score_col in df.columns:
                cols_to_add.append(score_col)

            component_cols.extend(cols_to_add)

        column_order = first_cols + component_cols + ["Overall Score"]

        # Reorder columns and keep only those present in the DataFrame
        columns_to_use = [col for col in column_order if col in df.columns]
        df = df[columns_to_use]

        # Sort by entry_id, pairID
        df.entry_id = df.entry_id.astype(int)
        df = df.sort_values(by=["entry_id", "PairID"])

        return df

    def _transpose_df(self, df: pd.DataFrame, prefix: str = '', drop_index: bool = True) -> pd.DataFrame:
        """Transpose a dataframe and make the first row column header"""
        df = df.T.reset_index()
        df.columns = [f'{prefix} {k}' if k != 'task_type' else k for k in df.iloc[0]]
        df = df[1:]
        df = df.reset_index(drop=drop_index)
        return df

    def create_task_summary_comparison_df(self, pretrained_dfdict: dict, sft_dfdict: dict) -> pd.DataFrame:
        """Transpose pre-trained and SFT 'task_summary_df' and concatename them side by side into one dataframe"""
        # Example:  task_summary_comparison_df = create_task_summary_comparison_df(pretrained_dfdict, sft_dfdict)

        df_list = []
        for prefix, df in zip(['pre_trained', 'SFT'],
                              [pretrained_dfdict['task_summary_df'].copy(), sft_dfdict['task_summary_df'].copy()]):
            df_list.append(self._transpose_df(df, prefix=prefix, drop_index=False))

        finaldf = df_list[0].merge(df_list[1], how='outer', on='task_type', sort=False,
                                   validate='one_to_one', indicator=True)

        if set(finaldf._merge) == {'both'} and set(finaldf.index_x == finaldf.index_y) == {True}:
            finaldf = finaldf.sort_values(by='index_x')
            finaldf = finaldf.drop(columns=['_merge', 'index_x', 'index_y'])

        return finaldf

    """# ===========================================================================
    # Core evaluation workflow function
    # ==========================================================================="""

    def evaluate(self, input_dict: Optional[Dict[int, Dict[str, list]]], show_tables: bool = False,
            return_input_dict: bool = False) -> dict:
        """
        Main evaluation function. Takes a nested dict of pred-gold pairs, output a dict of evaluation result dicts and dataframes.
        Args:
            input_dict (dict):  E.g. `{1: {'label': ['<asp>...</asp>', '<opn>..</opn>'], 'pred': ['<asp>..</asp>', '<opn>...</opn>'], 'task_type': 'AOPE'}, 2:...}`
            show_tables (bool): If true, display the dataframes in the output (while also being returned)
            return_input_dict (bool): If true, add the input_dict into returned results

        Returns:
            results (dict): a dict of eval results.

            results = {"dataframes": {
                            "entries_metrics_df": entries_metrics_df,
                            "pairs_metrics_df": pairs_metrics_df,
                            "task_summary_df": task_summary_df,
                            "match_details_df": match_details_df
                            },
                        "match_details_dict": match_details
                        }
        """

        # Input validation
        if not isinstance(input_dict, dict) or not input_dict:
            raise ValueError("Input must be a non-empty dictionary")

        for entry_id, entry_data in input_dict.items():
            required_keys = ['pred', 'label']
            if not all(key in entry_data for key in required_keys):
                raise ValueError(f"Entry {entry_id} is missing required keys: {required_keys}")

        entries_metrics = []
        pairs_metrics = []
        task_summary_data = defaultdict(list)
        match_details = {}

        # =========== Process each entry ==================
        for entry_id, entry_data in input_dict.items():
            # Extract data
            pred = entry_data.get('pred', [])  # list of strings, e.g. ['<asp>...</asp>', '<opn>..</opn>']
            gold = entry_data.get('label', [])  # same as pred
            task_type = entry_data.get('task_type', None)  # string, one of 'OE', 'AOPE', 'AOC', 'ASTE', 'ASQE', 'SC'.
            input_text = entry_data.get('input_text',
                                        '')  # for providing info in output, not part of metrics computation

            # Parse items, e.g. pred_items = [{'aspect': '...', 'category': '...', 'opinion': '...'}, {...}]
            pred_items, pred_task_type = self.parse_items(pred, task_type)
            gold_items, gold_task_type = self.parse_items(gold, task_type)

            # Determine task type with validation
            if task_type is None:
                if pred_task_type != gold_task_type and pred_items and gold_items:
                    raise ValueError(
                        f"Task type mismatch: predictions are {pred_task_type}, gold standard is {gold_task_type}")
                task_type = gold_task_type if gold_task_type != "unknown" else pred_task_type

            if task_type == "unknown":
                raise ValueError(f"Could not determine task type for entry {entry_id}")

            # Handle empty cases
            if len(pred_items) == 0 or len(gold_items) == 0:
                entry_metric = self._create_empty_metrics(entry_id, task_type, len(pred_items), len(gold_items))

                # Add input_text, gold, pred columns
                entry_metric['input_text'] = input_text
                entry_metric['gold'] = gold
                entry_metric['pred'] = pred

                entries_metrics.append(entry_metric)
                task_summary_data[task_type].append(entry_metric)

                # Empty pair metric
                pairs_metrics.append({
                    "entry_id": entry_id,
                    "task_type": task_type,
                    "pair_id": f"{entry_id}-g_-p_",
                    "gold_item": None,
                    "pred_item": None,
                    "weighted_score": 0.0
                })

                # Empty match detail
                match_details[entry_id] = {
                    "task_type": task_type,
                    "gold_items": gold_items,
                    "pred_items": pred_items,
                    "bipartite_indices": (np.array([]), np.array([])),
                    "input_text": input_text,
                    "matches": []
                }

                continue

            # ---------------------------------------------------------------
            # Basic unit of metrics - within entry_id, pair-level match score
            # ---------------------------------------------------------------
            # Calculate similarities and find optimal 1:1 matching for each pred-gold pair per entry_id
            # component_similarities_dict contains all possible matches and their component scores for this entry_id.
            # E.g. for AOC task where gold has 2 str units, pred has 2 str units:  component_similarities_dict = {
            #     (0, 0): {'aspect': 1.0, 'category': 0.1, 'opinion': 1.0},
            #     (0, 1): {'aspect': 0.0, 'category': 0.1, 'opinion': 0.0},
            #     (1, 0): {'aspect': 1.0, 'category': 1.0, 'opinion': 0.0},
            #     (1, 1): {'aspect': 0.0, 'category': 1.0, 'opinion': 1.0}
            #     }
            # similarity_matrix, row_ind, col_ind, component_similarities_dict = self.bipartite_matching(pred_items, gold_items, task_type)

            # Calculate similarities (pass input_text for calculating aspect/opinon similarity score)
            similarity_matrix, row_ind, col_ind, component_similarities_dict = self.bipartite_matching(pred_items,
                                                                                                       gold_items,
                                                                                                       task_type,
                                                                                                       input_text)

            # Get matched pairs as a list of tuples of (pred_idx, gold_idx),
            # The example above gives 2 matched pairs: matched_pairs = [(0, 0), (1, 1)]
            matched_pairs = [(row_ind[i], col_ind[i]) for i in range(len(row_ind))
                             if row_ind[i] < len(pred_items) and col_ind[i] < len(gold_items)]

            # ---------------------------------------------------------------
            # For output - match_details (dict and df)
            # ---------------------------------------------------------------
            # Entry-level details on matched pairs.  See Example above this function
            # contains all candidate pairs and whether they are optimal_match.
            # For each pair, contains component_score (similarity score) for the pair (e.g. {'aspect': 0.0, 'opinion': 0.5, 'category': 0.3}

            # 1) Get TP, FP, FN, TN (and matches, scores) per each entry_id (from pair scores)
            # pair-level component similarity score + threshold ==> TP, FP, TN, FN counts for the entry_id
            # component_stats = self.calculate_component_stats(
            #     pred_items, gold_items, matched_pairs, task_type, component_similarities_dict
            # )

            # Calculate component stats with input_text
            component_stats = self.calculate_component_stats(
                pred_items, gold_items, matched_pairs, task_type, component_similarities_dict, input_text
            )

            # Extract unit match status from component_stats
            unit_match_status = component_stats.get("_unit_match_status", {})

            entry_match_detail = {
                "task_type": task_type,
                "gold_items": gold_items,
                "pred_items": pred_items,
                "bipartite_indices": (row_ind, col_ind),
                "input_text": input_text,
                "matches": self._create_match_information(
                    pred_items, gold_items, row_ind, col_ind,
                    similarity_matrix, component_similarities_dict,
                    unit_match_status
                )
            }

            match_details[entry_id] = entry_match_detail

            # ---------------------------------------------------------------
            # For output - pair_metrics df
            # ---------------------------------------------------------------
            # Create pair metrics directly from precomputed data
            # Updates 'pairs_metrics' list
            self._create_pair_metrics(entry_id, task_type, matched_pairs, component_similarities_dict,
                                      similarity_matrix, pred, gold, pairs_metrics,
                                      unit_match_status
                                      )

            # ---------------------------------------------------------------
            # Compute entry_id level metric stats (by pair component scores)
            # ---------------------------------------------------------------
            # Create entry metrics (precision, recall, f1(s))
            entry_metric = self._create_entry_metrics(
                entry_id, task_type, matched_pairs, pred_items, gold_items, component_stats
            )

            # Add input_text, gold, pred columns
            entry_metric['input_text'] = input_text
            entry_metric['gold'] = gold
            entry_metric['pred'] = pred

            entries_metrics.append(entry_metric)
            task_summary_data[task_type].append(entry_metric)

        # =========== Above entry-level ==================
        # ---------------------------------------------------------------
        # For output - compute aggregated task_type-level metrics
        # ---------------------------------------------------------------
        # Calculate task-level metrics
        task_summary = self._calculate_task_summary(task_summary_data)

        # ---------------------------------------------------------------
        # Prepare output
        # ---------------------------------------------------------------
        # Create DataFrames
        entries_metrics_df = pd.DataFrame(entries_metrics)
        pairs_metrics_df = pd.DataFrame(pairs_metrics)
        task_summary_df = pd.DataFrame(task_summary)
        match_details_df = self.create_match_df(match_details)

        # Reorder columns to put components in the specified order
        entries_metrics_df = self._reorder_columns(entries_metrics_df)
        pairs_metrics_df = self._reorder_columns(pairs_metrics_df)
        task_summary_df = self._reorder_columns(task_summary_df)

        # Display tables if requested
        if show_tables:
            try:
                print("\nTask Type Summary:")
                display(task_summary_df)

                print("\nEntry-level Metrics:")
                display(entries_metrics_df)

                print("\nMatched-pair-level Metrics:")
                display(pairs_metrics_df)

                print("\nAll match Details:")
                display(match_details_df)
            except Exception as e:
                print(f"\nWarning: Could not display tables: {e}")
                print("Install IPython or use in a Jupyter notebook to display tables.")

        results = {
            "dataframes": {
                "task_summary_df": task_summary_df,
                "entries_metrics_df": entries_metrics_df,
                "pairs_metrics_df": pairs_metrics_df,
                "match_details_df": match_details_df
            },
            "match_details_dict": match_details
        }

        # Add input dict if requested
        if return_input_dict:
            results["input_dict"] = input_dict

        return results

    """# ===========================================================================
    # Main entry points for model evaluation
    # ==========================================================================="""

    def evaluate_model(self, pipeline, generation_args, test_dataset, view_result=False, view_size=10,
            handle_format_error=False, show_table=False, return_input_dict=False, show_first_5_messages=False) -> dict:
        """Evaluate models via Huggingface Pipeline
        Args:
            pipeline (Pipelines): An instance of Transformers.Pipelines
            generation_args (dict): args such as temperature, max new token
            test_dataset (Dataset): Huggingface Dataset object containing the test data, including columns/fields: 'task_type', 'user_content', 'ground_truth', 'input_text'.
            view_result (bool): Whether to  print the view_size number of input and generated output for eye-balling
            view_size (int): The number of input and model output pairs to display for eye-balling
            handle_format_error (bool): # If False, incorrect output format, e.g. additional characters around array '[', ']' will not be handled.
            show_table (bool): If True, will display evaluation result statistics tables
            return_input_dict (bool): If True, the function output will include a python dict containing the test input and model generation output
            show_first_5_messages (bool): If True, print the first 5 message arrays for debugging.

        Returns:
             A dict of eval results containing {"all_preds": all_preds, "all_labels": all_labels, "eval_input": eval_input, "results": results},
             where  results = {"dataframes": {
                            "entries_metrics_df": entries_metrics_df,
                            "pairs_metrics_df": pairs_metrics_df,
                            "task_summary_df": task_summary_df,
                            "match_details_df": match_details_df
                            },
                        "match_details_dict": match_details
                        }
        """

        all_preds = []
        all_labels = []
        eval_input = dict()
        view_size = None if not view_result else len(test_dataset) if not view_size else min(view_size,
                                                                                             len(test_dataset))

        for idx, sample in enumerate(tqdm(test_dataset, desc="Evaluating")):

            sample_field_dict = self.format_user_content(sample)

            ground_truth = sample_field_dict['ground_truth']
            task_type = sample_field_dict['task_type']
            input_text = sample_field_dict['input_text']

            system_prompt = "You are an expert in aspect-based sentiment analysis."
            user_content = sample_field_dict['user_content']

            messages = [{"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content}]

            if show_first_5_messages and idx < 5:
                print(f'\033[36m{idx} ({task_type}): \033[0m {messages}')

            raw_prediction = pipeline(messages, **generation_args)[0]['generated_text']
            prediction = self.process_pred_str(raw_prediction, handle_format_error=handle_format_error)

            all_preds.append(prediction)
            all_labels.append(ground_truth)

            eval_input[idx] = {'task_type': task_type, 'input_text': input_text, 'label': ground_truth,
                               'pred': prediction}

            if view_size and idx < view_size:
                input_text = sample_field_dict['input_text']
                dash_line = f"\n{'-' * 100}\n"
                raw_pred_str = {raw_prediction}
                print(
                    f"{idx}. \033[1;30;44m{task_type:^10}\033[0m")  # 1; (bold)  30; (black font) ^10 (centre, width = 10)
                print(f"Input:\n\033[33m{input_text}\033[0m")
                print(f"ground_truth:\033[32m\n{ground_truth}\033[0m")
                print(f"raw_pred:\033[35m\n{raw_pred_str}\033[0m")
                print(f"prediction:\033[36m\n{prediction}{dash_line}\033[0m")

        if type(all_preds) == type(all_labels) and type(all_preds[0]) == type(all_labels[0]):
            ####################################################
            # Evaluate results
            results = self.evaluate(eval_input, show_tables=show_table, return_input_dict=return_input_dict)
            ####################################################

            return {"all_preds": all_preds, "all_labels": all_labels, "eval_input": eval_input, "results": results}
        else:
            print(
                f'type(all_preds): {type(all_preds)}\ntype(all_labels): {type(all_labels)}\ntype(all_preds[0]): {type(all_preds[0])}\ntype(all_labels[0]): {type(all_labels[0])}')
            raise ValueError("Check prediction data structure to ensure it matches with the labels")

    def evaluate_from_saved_dict(self, eval_input_dict, show_tables=False, return_input_dict=False) -> dict:
        """
        Evaluate from a pre-saved eval_input dictionary (e.g., loaded from a .py file).

        Args:
            eval_input_dict (dict): Dictionary with structure {entry_id: {'task_type': ..., 'input_text': ..., 'label': [...], 'pred': [...]}}
            show_tables (bool): If True, display evaluation result statistics tables
            return_input_dict (bool): If True, include the eval_input_dict in returned results

        Returns:
            dict: Same format as evaluate_model output:
                {
                    "all_preds": all_preds,
                    "all_labels": all_labels,
                    "eval_input": eval_input_dict,
                    "results": {
                        "dataframes": {
                            "task_summary_df": ...,
                            "entries_metrics_df": ...,
                            "pairs_metrics_df": ...,
                            "match_details_df": ...
                        },
                        "match_details_dict": ...
                    }
                }

        Example:
            from saved_predictions import sft_eval_input  # Load from .py file
            output = evaluator.evaluate_from_saved_dict(sft_eval_input, show_tables=True)
            task_summary_df = output["results"]["dataframes"]["task_summary_df"]
        """
        # Extract all_preds and all_labels from eval_input_dict
        all_preds = []
        all_labels = []

        # Sort by entry_id to maintain order
        sorted_entries = sorted(eval_input_dict.items(), key=lambda x: x[0])

        for entry_id, entry_data in sorted_entries:
            all_preds.append(entry_data['pred'])
            all_labels.append(entry_data['label'])

        # Call the main evaluate function
        results = self.evaluate(
            eval_input_dict,
            show_tables=show_tables,
            return_input_dict=return_input_dict
        )

        # Return in the same format as evaluate_model
        return {
            "all_preds": all_preds,
            "all_labels": all_labels,
            "eval_input": eval_input_dict,
            "results": results
        }
