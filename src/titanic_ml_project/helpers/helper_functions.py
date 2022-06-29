import re
from importlib import import_module

def model_from_string(
    model_name
):
    return getattr(
        import_module(
            ('.').join(model_name.split('.')[:-1])
        ),
        model_name.rsplit(".")[-1],
    )()

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
