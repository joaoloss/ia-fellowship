"""
Module for resolving the type of a given string value.
Current supported types: number, date, email, string.
"""

import re
from datetime import datetime
class TypeResolver:
    def __init__(self):
        self.__types = ["date", "number", "email", "string"] # Order matters for resolution

    def date(self, value: str) -> bool:
        POSSIBLE_DATE_FORMATS = [
            "%d/%m/%Y", "%d/%m/%y",
            "%d-%m-%Y", "%d-%m-%y",
            "%Y-%m-%d", "%Y/%m/%d",
            "%m/%d/%Y", "%m/%d/%y",
            "%m-%d-%Y", "%m-%d-%y"
        ]
        """
        Check if the value can be parsed as a date.
        Accepts various date formats.
        """
        try:
            for fmt in POSSIBLE_DATE_FORMATS:
                try:
                    datetime.strptime(value, fmt)
                    return True
                except ValueError:
                    continue
        except ValueError:
            return False
    
    def number(self, value: str) -> bool:
        """
        Check if the value can be interpreted as a number.
        Accepts integers and floats with optional commas, dots and spaces.
        """
        
        count_digits = len(re.findall(r'\d', value))
        if count_digits == 0:
            return False
        if (count_digits / len(value)) >= 0.65: # at least 65% digits
            return True
        return False

    def string(self, value: str) -> bool:
        """
        All values can be considered strings. It's the fallback type.
        """
        return True

    def email(self, value: str) -> bool:
        """
        Check if the value is a valid email address.
        """
        email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(email_regex, value) is not None

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