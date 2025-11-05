"""
Module for resolving the type of a given string value.
Current supported types: number, date, string.
"""

import re
from dateutil.parser import parse as parse_date

class TypeResolver:
    def __init__(self):
        self.__types = ["number", "date", "string"] # Order matters for resolution

    def date(self, value: str) -> bool:
        """
        Check if the value can be parsed as a date.
        Accepts various date formats.
        """
        try:
            parse_date(value, dayfirst=True)
            return True
        except (ValueError, OverflowError):
            return False
    
    def number(self, value: str) -> bool:
        """
        Check if the value can be interpreted as a number.
        Accepts integers and floats with optional commas, dots and spaces.
        """
        
        filtered = value.replace(",", "", count=1).replace(" ", "").replace(".", "", count=1)
        if filtered.isdigit():
            return True
        return False

    def string(self, value: str) -> bool:
        """
        All values can be considered strings. It's the fallback type.
        """
        return True

    def resolve(self, value: str) -> str | None:
        """
        Resolve the type of the given value.
        Returns the type name as a string or None if no type matches.
        """
        
        if not value: # empty value or None
            return None
        
        for type_name in self.__types:
            method = getattr(self, type_name, None)
            if method and method(value):
                return type_name
        return None # should not reach here due to string being the fallback type