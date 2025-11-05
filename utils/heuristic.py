"""
This module implements a heuristic caching mechanism to optimize data extraction from PDF documents.
It maintains a cache of heuristics based on previously extracted data to improve the accuracy and efficiency
of future extractions. The heuristics are applied during preprocessing to fill in fields in the request schema
based on cached data, and the cache is updated with new observations after each extraction.
"""

from copy import deepcopy
from typing import Dict, List
import logging

from utils.type_resolution import TypeResolver
from utils.pdf2mat import PDF2Matrix

# Logging setup
logger = logging.getLogger("my_logger")

class Heuristic:
    def __init__(self, num_heuristics_per_key: int = 5):
        """
        Initialize the Heuristic object with a specified number of maximum heuristics to store per key.
        """
        if num_heuristics_per_key < 1:
            raise ValueError("num_heuristics_per_key must be >= 1")
        
        self.num_heuristics_per_key = int(num_heuristics_per_key)
        self.cache: Dict[str, Dict[str, Dict]] = {}
        self.type_resolver = TypeResolver()

    def get_cache(self) -> Dict:
        """
        Return a deep copy of the current heuristic cache to avoid accidental mutation.
        """
        return deepcopy(self.cache)

    def heuristic_preprocessing(
        self,
        label: str,
        request_schema: Dict[str, dict],
        pdf_matrix_representation: List[List[str]],
    ) -> None:
        """
        Apply heuristic preprocessing to fill in fields in the request schema based on cached heuristics.
        Return a partial_result dict with filled fields.
        """

        if label not in self.cache: # No cached heuristics for this label
            return dict()
        
        partial_result = dict()

        for key in list(request_schema.keys()):

            cache_for_key = self.cache[label].get(key)
            if not cache_for_key: # No cached heuristics for this key
                continue

            # Search for matching heuristics
            for record_heuristic in cache_for_key["heuristics"]:
                position = record_heuristic.get("position")
                if not position:
                    continue

                try:
                    # 2D position (row, col)
                    if len(position) == 2:
                        row_index, col_index = position
                        pdf_element = pdf_matrix_representation[row_index][col_index]
                    elif len(position) == 1:
                        row_index = position[0]
                        pdf_element = " ".join(pdf_matrix_representation[row_index])
                    else:
                        # Unexpected shape
                        logger.debug(f"Unexpected position shape: {position}")
                        continue
                except (IndexError, TypeError) as e:
                    # Out of bounds or malformed matrix
                    logger.debug(f"Position lookup failed for {key} at {position}: {e}")
                    continue

                resolved_type = self.type_resolver.resolve(pdf_element)
                if resolved_type != record_heuristic.get("type"):
                    continue

                # If type is string, check length similarity using stored mean_length
                # as string type is very generic and prone to false positives
                # if resolved_type == "string":
                #     mean_len = record_heuristic.get("mean_length")
                #     if mean_len is not None and mean_len > 0:
                #         ratio = len(pdf_element) / mean_len

                #         # Skip heuristic if length differs too much (>30%)
                #         if abs(1.0 - ratio) > 0.30:
                #             logger.debug(f"Length mismatch for {key}: {len(pdf_element)} vs mean {mean_len:.2f}")
                #             continue
                    
                #     # Update mean_length with new sample
                #     sample_count = record_heuristic.get("match_count", 0)
                #     new_mean_length = (mean_len * sample_count + len(pdf_element)) / (sample_count + 1) if mean_len is not None else len(pdf_element)
                #     record_heuristic["mean_length"] = new_mean_length

                # Accept heuristic match
                partial_result[key] = pdf_element
                record_heuristic["match_count"] = record_heuristic.get("match_count", 0) + 1

                break  # Stop trying other heuristics for this key
            else:
                logger.debug(f"No matching heuristic found for key {key} under label {label}")
        
        return partial_result

    def heuristic_update(self, result: Dict[str, str], label: str, pdf_matrix: PDF2Matrix) -> None:
        """
        Update the heuristic cache with observed partial_result values.
        """
        if not result or label is None:
            return

        if label not in self.cache:
            self.cache[label] = dict()

        for key, value in result.items():
            if not value: # Skip empty or null values
                continue

            pdf_element_position = pdf_matrix.get_position_of_text(value)
            if pdf_element_position is None:
                continue

            value_type = self.type_resolver.resolve(value)

            # Heuristic record includes mean_length and sample_count for robust length checks
            heuristic_definition = {
                "type": value_type,
                "position": pdf_element_position,
                "match_count": 1,
            }

            # If string, store length stats
            if value_type == "string":
                heuristic_definition["mean_length"] = len(value)
                heuristic_definition["sample_count"] = 1

            if key not in self.cache[label]: # New key for this label
                self.cache[label][key] = {
                    "count": 1,
                    "heuristics": [heuristic_definition],
                }
                continue

            # Merge into existing heuristics: if same position+type exists, update stats
            heuristics = self.cache[label][key]["heuristics"]
            merged = False
            for rec in heuristics:
                if rec.get("position") == pdf_element_position and rec.get("type") == value_type:
                    # update match_count and length stats if present
                    rec["match_count"] = rec.get("match_count", 0) + 1
                    if value_type == "string":
                        prev_mean = rec.get("mean_length", 0)
                        prev_samples = rec.get("sample_count", 0)
                        new_samples = prev_samples + 1
                        new_mean = (prev_mean * prev_samples + len(value)) / new_samples if new_samples else len(value)
                        rec["mean_length"] = new_mean
                        rec["sample_count"] = new_samples
                    merged = True
                    break

            if not merged:
                heuristics.append(heuristic_definition)

            # Keep top heuristics by match_count
            heuristics.sort(key=lambda x: x.get("match_count", 0), reverse=True)
            self.cache[label][key]["heuristics"] = heuristics[:self.num_heuristics_per_key]
            self.cache[label][key]["count"] = self.cache[label][key].get("count", 0) + 1