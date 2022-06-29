import re
import pandas as pd

def get_title(
    name: str
) -> str:
    """
    This function uses regex to get Titles out of
    Peoples name in Titanic.
    """
    title_search = re.search(' ([A-Za-z]+)\.', name)

    if title_search:
        return title_search.group(1)
    return ""
