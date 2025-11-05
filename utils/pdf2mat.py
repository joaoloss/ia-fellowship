"""
This module provides a class to convert PDF documents into a matrix representation
based on the spatial arrangement of text boxes. It also includes functionality to
locate the position of specific text within the matrix. Only horizontal text boxes are considered.
"""

from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextBoxHorizontal
import re
import editdistance

class PDF2Matrix:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path

    def create_matrix_representation(self) -> list[list[str]]:
        """
        Convert the PDF into a matrix representation based on text box positions.
        Returns a 2D list (matrix) where each sublist represents a row of text.
        All text is converted to lowercase and stripped of extra whitespace and the spaces are normalized.
        """
        boxes = self.__extract_text_boxes_split()
        rows = self.__group_into_rows(boxes)
        rows = self.__sort_row_items(rows)
        self.__pdf_mat = self.__rows_to_matrix(rows)
        return self.__pdf_mat

    def get_position_of_text(self, text: str) -> tuple | None:
        """
        Locate the position of the specified text in the PDF matrix.
        Returns a tuple if found, otherwise None.

        The tuple format is:
        - (row,) if the text matches an entire row
        - (row, col) if the text matches a specific cell within a row
        """

        if not self.__pdf_mat or not text:
            return None

        text = re.sub(r"\s+", " ", text.strip().lower()) # basic preprocessing - the same as during matrix creation
        for row_index, row in enumerate(self.__pdf_mat):
            if len(row) == 1:
                if row[0] == text:
                    return (row_index,)  # Return the position as (row,)

            row_str = " ".join(row).lower()
            if len(row_str) <= 10:
                if row_str == text:
                    return (row_index,)  # Return the position as (row,)
                
            # If the row is longer than 10 characters, allow fuzzy matching
            elif editdistance.eval(row_str, text) / max(len(row_str), len(text)) < 0.10: # fuzzy match
                return (row_index,)  # Return the position as (row,)

            # Find exact match within the row
            for col_index, col in enumerate(row):
                if col == text:
                    return (row_index, col_index)  # Return the position as (row, col)
        
        return None
            
    def __extract_text_boxes_split(self):
        boxes = list()
        for page_layout in extract_pages(self.pdf_path):
            for element in page_layout:
                if isinstance(element, LTTextBoxHorizontal):
                    x0, y0, x1, y1 = element.x0, element.y0, element.x1, element.y1
                    text = re.sub(r"\s+", " ", element.get_text().strip().lower())  # basic preprocessing
                    if text:  # only consider non-empty text boxes
                        boxes.append({
                            "text": text,
                            "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                            "cx": (x0 + x1) / 2,
                            "cy": (y0 + y1) / 2,
                        })
        return boxes

    def __group_into_rows(self, boxes, y_threshold=20):
        rows = list()
        for box in sorted(boxes, key=lambda b: -b["cy"]):  # top to bottom
            for row in rows:
                if abs(row["cy"] - box["cy"]) < y_threshold:
                    row["items"].append(box)
                    break
            else: # not placed in any existing row
                rows.append({"cy": box["cy"], "items": [box]})
        
        return rows

    def __sort_row_items(self, rows):
        for row in rows:
            row["items"] = sorted(row["items"], key=lambda b: b["cx"])
        return rows

    def __rows_to_matrix(self, rows):
        matrix = list()
        for row in rows:
            matrix.append([item["text"] for item in row["items"]])
        return matrix
