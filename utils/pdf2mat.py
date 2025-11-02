from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextBoxHorizontal
import os
import json

class PDF2Matrix:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
    
    def get_matrix(self):
        boxes = self.__extract_text_boxes_split()
        rows = self.__group_into_rows(boxes)
        rows = self.__sort_row_items(rows)
        pdf_mat = self.__rows_to_matrix(rows)
        return pdf_mat

    def __extract_text_boxes_split(self):
        boxes = list()
        for page_layout in extract_pages(self.pdf_path):
            for element in page_layout:
                if isinstance(element, LTTextBoxHorizontal):
                    x0, y0, x1, y1 = element.x0, element.y0, element.x1, element.y1
                    text = element.get_text().strip().lower()
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
