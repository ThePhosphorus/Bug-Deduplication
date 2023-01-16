import importlib
from datetime import datetime, timezone

from data.bug_report_database import BugReport

def load_class(tokenizer_path: str) -> object:
    """
    Load tokenizer from a string.
    :param tokenizer_path: module_name.class_name
    :return: object
    """
    module_name, class_name = tokenizer_path.rsplit('.', 1)
    my_module = importlib.import_module(module_name)
    return getattr(my_module, class_name)()


def read_date_from_report(bug: BugReport) -> datetime:
    return  datetime.fromtimestamp(bug.creation_ts, tz=timezone.utc) if (bug.creation_ts) else datetime.now()
