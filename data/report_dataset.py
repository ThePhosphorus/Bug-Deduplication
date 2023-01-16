"""
Each dataset has bug report ids and the ids of duplicate bug reports.
"""
from __future__ import annotations
from typing import List,Tuple, Optional, Generator

from data.bug_report_database import BugReport, BugReportDatabase

class ReportDataset(object):

    def __init__(self, file: Optional[str] = None, bug_Ids: Optional[List[int]] = None, duplicateIds: Optional[List[int]] = None, ts: Optional[Tuple[int,int]] =None  ):
        if file is not None:
            f = open(file, 'r')
            self.info = f.readline().strip()

            self.bugIds = [int(id) for id in f.readline().strip().split()]
            self.duplicateIds = [int(id)
                                 for id in f.readline().strip().split()]
        else:
            self.bugIds = bug_Ids if bug_Ids else []
            self.duplicateIds = duplicateIds if duplicateIds else []
            self.start_ts = ts[0] if ts else 0
            self.end_ts = ts[1] if ts else 0

    @staticmethod
    def split(report_db: BugReportDatabase) -> Tuple[ReportDataset, ReportDataset, ReportDataset]:
        # get reports by date
        report_by_date: List[Tuple[int, int]] = [(report.bug_id, report.creation_ts) for report in report_db]
        report_by_date = sorted(report_by_date, key=lambda x: x[1], reverse=False) # sort from oldest to newest

        min_date = report_by_date[0][1]
        # max_date = report_by_date[-1][1]

        day_to_sec = 24 * 3600

        # Split into warmup (10%), training (60%), validation (10%) and test(20%)
        warm_date_range = (min_date,  min_date+ 350 * day_to_sec)
        train_date_range = (warm_date_range[1],  warm_date_range[1]+ 3850 * day_to_sec)
        val_date_range = (train_date_range[1],  train_date_range[1]+ 140 * day_to_sec)
        test_date_range = (val_date_range[1],  val_date_range[1] + 700 * day_to_sec)

        split_dates: Tuple[int, int, int] = (test_date_range, val_date_range, train_date_range)
        bugs: Tuple[List[int],List[int],List[int]] = ([],[],[])
        dups: Tuple[List[int],List[int],List[int]] = ([],[],[])  

        for report in report_db:
            ts : int = report.creation_ts
            for i, (split_date, split_date_end) in enumerate(split_dates):
                if split_date <= ts <= split_date_end :
                    if report.dup_id is not None:
                        dups[i].append(report.bug_id)
                    else :
                        bugs[i].append(report.bug_id)
                    break
        
        # Return Train, Val, Test
        return (ReportDataset(file=None,bug_Ids=bugs[2],duplicateIds=dups[2], ts=train_date_range), \
                ReportDataset(file=None,bug_Ids=bugs[1],duplicateIds=dups[1], ts=val_date_range), \
                ReportDataset(file=None,bug_Ids=bugs[0],duplicateIds=dups[0], ts=test_date_range))

    
    @staticmethod
    def progressive(report_db: BugReportDatabase) -> Generator[ReportDataset, None, None] :
        # Get all dups
        duplicates : List[Tuple[int, BugReport]] = [ (idx, report) for idx, report in enumerate(report_db.report_list) if report.dup_id is not None ]

        increment : int = len(duplicates) // 50

        current_pos = 0
        for i in range(increment, len(duplicates), increment) :
            pos, dup = duplicates[i]
            report_db.load_until(dup.creation_ts)
            span = (report_db.report_list[current_pos].creation_ts,report_db.report_list[pos + 1].creation_ts )
            yield ReportDataset(file=None, bug_Ids=[report.bug_id for report in report_db.report_list[current_pos:pos + 1]], duplicateIds= [report.bug_id for report in report_db.report_list[current_pos:pos + 1] if report.dup_id is not None], ts=span)
            current_pos = pos + 1

        if current_pos < len(report_db.report_list) :
            span = (report_db.report_list[current_pos].creation_ts,report_db.report_list[-1].creation_ts )
            yield ReportDataset(file=None, bug_Ids=[report.bug_id for report in report_db.report_list[current_pos:]], duplicateIds= [report.bug_id for report in report_db.report_list[current_pos:] if report.dup_id is not None], ts=span)            
