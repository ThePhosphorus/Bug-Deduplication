from xmlrpc.client import MAXINT
import os
import json
import codecs

from typing import Iterable, List, Dict, Set
from data.bug_report_database import BugReport, BugReportDatabase, Frame, StackTrace


ROOT_DIR: str = os.path.realpath(
    os.path.join(os.path.dirname(__file__), "../../"))
REPORT_LIST_TYPES: List[str] = ["training", "validation", "test"]
DATASETS_DIR: str = os.path.join(ROOT_DIR, "datasets")


class TestDatasetFolder():
    def test_true(self) -> None:
        assert True
    # Check if all datasets used for testing are in the right folders

    def test_datasets_location(self):
        # Check if the dataset folder is present
        assert os.path.exists(DATASETS_DIR)
        assert os.path.isdir(DATASETS_DIR)

    def test_campbell_dataset_locations(self):
        # Check if the containing folder exists
        campbell_dataset_dir: str = os.path.join(
            DATASETS_DIR, "campbell_dataset")
        assert os.path.exists(campbell_dataset_dir)
        assert os.path.isdir(campbell_dataset_dir)
        # Check if reports dataset exists
        reports_dataset: str = os.path.join(
            campbell_dataset_dir, "reports.json")
        assert os.path.exists(reports_dataset)
        # Check if reports dataset can be read in json
        with codecs.open(reports_dataset, 'r') as file:
            assert json.load(file)
        # Test the presence of report lists
        for list_type in REPORT_LIST_TYPES:
            assert os.path.exists(os.path.join(
                campbell_dataset_dir, list_type + "_campbell.txt"))


# Testing BugReportDatabase
class TestBugReportDatabase():
    def test_true(self) -> None:
        assert True
    def test_remove_recursion(self) -> None:
        RANGE: Iterable[int] = range(30000)

        stacktraces: List[List[Frame]] = [
            [Frame(0, 'A'), Frame(0, 'B'), Frame(0, 'C'), Frame(0, 'D'), Frame(0, 'B'), Frame(0, 'C'), Frame(0, 'B'), Frame(0, 'E'), Frame(0, 'F'), Frame(0, 'B'), Frame(0, 'C'), Frame(0, 'D')],
            [Frame(0, 'A') for _ in RANGE],
            [ f for _ in RANGE for f in [Frame(0, A, Frame(0, 'B'), Frame(0, 'C')]],
            [Frame(0, 'A'), Frame(0, 'B'), *[f for _ in RANGE for f in [Frame(0, 'A'), Frame(0, 'B'), Frame(0, 'C')]], Frame(0, 'D')],
            [Frame(0, 'A'), *[f for _ in RANGE for f in [Frame(0, 'B'), Frame(0, 'C')]]],
            [Frame(0, 'A'), *[Frame(0, 'B') for _ in RANGE]],
            [*[f for _ in RANGE for f in [Frame(0, 'A'), Frame(0, 'B'), Frame(0, 'C'), Frame(0, 'A'), Frame(0, 'D'), Frame(0, 'B')]]],
            [Frame(0, 'A'),Frame(0, 'B'),Frame(0, 'C'),Frame(0, 'D'),Frame(0, 'E'),Frame(0, 'D'),Frame(0, 'E'),Frame(0, 'D'),Frame(0, 'E'),Frame(0, 'A'),Frame(0, 'B'),Frame(0, 'C'),Frame(0, 'D'),Frame(0, 'E'),Frame(0, 'D'),Frame(0, 'E'),Frame(0, 'F')],
            [Frame(0, 'A'),Frame(0, 'B'),*[f for _ in RANGE for f in [Frame(0, 'C'),Frame(0, 'D')]],Frame(0, 'E'),*[f for _ in RANGE for f in [Frame(0, 'C'),Frame(0, 'D')]]],
            [Frame(0, 'A'), *[f for _ in RANGE for f in [Frame(0, 'B'), Frame(0, 'C')]], Frame(0, 'D'), *[f for _ in RANGE for f in [Frame(0, 'B'), Frame(0, 'C')]]],
            [Frame(0, 'A'), Frame(0, 'B'), Frame(0, 'C'), Frame(0, 'A'), Frame(0, 'D'), Frame(0, 'B')],
        ]

        expected_stacktraces: List[List[Frame]] = [
            [Frame(0, 'A'), Frame(0, 'B'), Frame(0, 'C'), Frame(0, 'D'), Frame(0, 'B'), Frame(0, 'C'), Frame(0, 'B'), Frame(0, 'E'), Frame(0, 'F'), Frame(0, 'B'), Frame(0, 'C'), Frame(0, 'D')],
            [Frame(0, 'A')],
            [Frame(0, 'A'), Frame(0, 'B'), Frame(0, 'C')],
            [Frame(0, 'A'), Frame(0, 'B'), Frame(0, 'C'), Frame(0, 'D')],
            [Frame(0, 'A'), Frame(0, 'B'), Frame(0, 'C')],
            [Frame(0, 'A'), Frame(0, 'B')],
            [Frame(0, 'A'), Frame(0, 'B'), Frame(0, 'C'), Frame(0, 'A'), Frame(0, 'D'), Frame(0, 'B')],
            [Frame(0, 'A'),Frame(0, 'B'),Frame(0, 'C'),Frame(0, 'D'),Frame(0, 'E'),Frame(0, 'F')],
            [Frame(0, 'A'),Frame(0, 'B'),Frame(0, 'C'),Frame(0, 'D'),Frame(0, 'E'),Frame(0, 'C'),Frame(0, 'D')],
            [Frame(0, 'A'), Frame(0, 'B'), Frame(0, 'C'), Frame(0, 'D'), Frame(0, 'B'), Frame(0, 'C')],
            [Frame(0, 'A'), Frame(0, 'B'), Frame(0, 'C'), Frame(0, 'A'), Frame(0, 'D'), Frame(0, 'B')],
        ]

        for stacktrace, expected in zip(stacktraces, expected_stacktraces):
            result = StackTrace.remove_recursion(stacktrace)
            assert expected == result

class TestBugReportDatabaseCampbell():
    dataset_dir: str = os.path.join(
        DATASETS_DIR, "campbell_dataset")

    def import_from_json(self, use_recursion: bool) -> BugReportDatabase:

        report_db: BugReportDatabase = BugReportDatabase.from_json(
            os.path.join(self.dataset_dir, "reports.json"), use_recursion)

        # Report list should not be empty
        assert len(report_db.report_list) > 0

        # Each bug report's stacktrace should have at least one frame
        for bug_report in report_db.report_list:
            for frames in bug_report.stacktrace.frames_lists :
                assert len(frames) > 0
                # no Frame should have a None str representation
                for frame in frames:
                    assert str(frame) != None

        return report_db

    def test_import_from_json_no_recursion(self) -> None:
        self.import_from_json(False)

    def test_import_from_json_with_recursion(self) -> None:
        report_db: BugReportDatabase = self.import_from_json(True)

        # Check if each bug report's stack report have repetition of one
        for bug_report in report_db.report_list:
            for frames in bug_report.stacktrace.frames_lists :
                past_frame: str = ""
                for frame in frames:
                    current_frame = str(frame)
                    assert past_frame != current_frame
                    past_frame = current_frame

    def test_master_report_being_dup_of_reports(self) -> None:
        # import cambell dataset
        report_db: BugReportDatabase = BugReportDatabase.from_json(
            os.path.join(self.dataset_dir, "reports.json"), False)

        master_by_report: Dict[int, int] = report_db.get_master_by_report(
            report_db.report_list)
        for report_id in master_by_report:
            master_id = master_by_report[report_id]
            report: BugReport = report_db.get_report(report_id)
            if report.dup_id is None:
                assert master_id == report_id
            else:
                assert master_id == report.dup_id

    def test_master_set_by_id(self) -> None:
        # import cambell dataset
        report_db: BugReportDatabase = BugReportDatabase.from_json(
            os.path.join(self.dataset_dir, "reports.json"), False)

        master_set_by_id: Dict[int, Set[int]] = report_db.get_master_set_by_id(
            report_db.report_list)

        for master_id in master_set_by_id:
            dup_set: Set[int] = master_set_by_id[master_id]
            for duplicate_id in dup_set:
                duplicate = report_db.get_report(duplicate_id)
                assert duplicate.dup_id == master_id or (
                    duplicate.dup_id == None and duplicate.bug_id == master_id)       

    def test_depth_order(self) -> None:
        # import cambell dataset
        report_db: BugReportDatabase = BugReportDatabase.from_json(
            os.path.join(self.dataset_dir, "reports.json"), False)

        for report in report_db:
            for frames in report.stacktrace.frames_lists:
                # Check depth order in stacktrace
                assert [frame.depth for frame in frames] == [frame.depth for frame in sorted(frames, key=lambda frame: frame.depth,reverse=False)]
    
    def test_stacks_sorted_by_longest(self) -> None:
        # import cambell dataset
        report_db: BugReportDatabase = BugReportDatabase.from_json(
            os.path.join(self.dataset_dir, "reports.json"), False)

        for report in report_db:
            past_length = MAXINT
            for frames in report.stacktrace.frames_lists:
                assert len(frames) <= past_length
                past_length = len(frames)
