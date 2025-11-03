"""
This module implements a heuristic caching mechanism to optimize data extraction from PDF documents.
It allows for storing and reusing heuristics based on previously extracted data to improve performance and accuracy.
It's currently based on positional and type matching within the PDF matrix representation.
"""

from utils.type_resolution import TypeResolver
from utils.pdf2mat import PDF2Matrix

class Heuristic:
    def __init__(self, num_heuristics_per_key: float = 5):
        self.num_heuristics_per_key = num_heuristics_per_key
        self.cache = dict()
        self.type_resolver = TypeResolver()

    def get_cache(self):
        """
        Return the current heuristic cache.
        """
        
        return self.cache

    def heuristic_preprocessing(self, label:str, request_schema:dict, partial_result:dict, pdf_matrix:list[list[str]]):
        """
        Apply heuristic preprocessing to fill in fields in the request schema based on cached heuristics.

        It checks for positional and type matches in the PDF matrix representation.
        If a match is found, the corresponding field in the partial result is filled and removed from the request schema.
        """

        if label not in self.cache: # No cached heuristics for this label
            return
        
        request_schema_keys = list(request_schema.keys())
        for key in request_schema_keys:
            if key not in self.cache[label]: # No cached heuristics for this key
                continue
            
            cache_for_key = self.cache[label][key]
            for record_heuristic in cache_for_key["heuristics"]:
                position = record_heuristic["position"]
                if position is None:
                    continue
                
                if len(position) == 2: # 2D position
                    row_index, col_index = position
                    try:
                        pdf_element = pdf_matrix[row_index][col_index]
                    except IndexError: # Not found field
                        continue

                    if record_heuristic["type"] == self.type_resolver.resolve(pdf_element):
                        # Position and type match: assume it is possible to fill the field directly via heuristic
                        partial_result[key] = pdf_element
                        record_heuristic["match_count"] += 1
                        request_schema.pop(key) # Remove key from request schema as it has been filled
                        break
                    else:
                        continue
                
                else: # 1D position
                    row = position[0]
                    try:
                        pdf_row = " ".join(pdf_matrix[row])
                    except IndexError: # Not found field
                        continue

                    if record_heuristic["type"] == self.type_resolver.resolve(pdf_row):
                        # Position and type match: assume it is possible to fill the field directly via heuristic
                        partial_result[key] = pdf_row
                        record_heuristic["match_count"] += 1
                        request_schema.pop(key) # Remove key from request schema as it has been filled
                        break
                    else:
                        continue
    
    def heuristic_update(self, partial_result:dict, label:str, pdf_matrix:PDF2Matrix):
        """
        Update the heuristic cache with new data from the partial result.
        """

        if label not in self.cache:
            self.cache[label] = dict()
        
        for key, value in partial_result.items():
            if not value: # Skip empty or null values
                continue
            
            pdf_element_position = pdf_matrix.get_position_of_text(value)
            if pdf_element_position is None:
                continue

            heuristic_definition = {
                "type": self.type_resolver.resolve(value),
                "position": pdf_element_position,
                "match_count": 1
            }

            if key not in self.cache[label]: # New key for this label
                self.cache[label][key] = {
                    "count": 0,
                    "heuristics": [heuristic_definition]
                }
            else:
                # Add new heuristic entry to an existing key
                self.cache[label][key]["heuristics"].append(heuristic_definition)

                # Keep only top self.num_heuristics_per_key heuristics per key based on match_count
                self.cache[label][key]["heuristics"] = sorted(self.cache[label][key]["heuristics"], key=lambda x: x["match_count"], reverse=True)[:self.num_heuristics_per_key]

            self.cache[label][key]["count"] += 1