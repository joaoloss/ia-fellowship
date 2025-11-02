import re
from dateutil.parser import parse as parse_date

class TypeResolver:
    def __init__(self):
        self.__types = ["number", "date", "string"]
    
    def date(self, value: str):
        try:
            parse_date(value, dayfirst=True)
            return True
        except (ValueError, OverflowError):
            return False
    
    def number(self, value: str):
        filtered = value.replace(",", "", count=1).replace(" ", "").replace(".", "", count=1)
        if filtered.isdigit():
            return True
        return False

    def string(self, value: str):
        return True

    def resolve(self, value: str):
        if not value: # empty value or None
            return None
        
        for type_name in self.__types:
            method = getattr(self, type_name, None)
            if method and method(value):
                return type_name
        return None