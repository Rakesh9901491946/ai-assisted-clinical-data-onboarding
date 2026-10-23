import pandas as pd
import sys
from types import SimpleNamespace

from onboarding import storage


def test_approved_output_uses_project1_raw_staging_contract(monkeypatch):
    statements = []
    class Cursor:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def executemany(self, sql, rows): statements.append((sql, list(rows)))
    class Connection:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def cursor(self): return Cursor()
    monkeypatch.setenv("OMOP_DATABASE_URL", "postgresql://example")
    # Import is lazy in the implementation; provide a mocked psycopg module.
    monkeypatch.setitem(sys.modules, "psycopg", SimpleNamespace(connect=lambda *_: Connection()))
    clean = pd.DataFrame([{ "person_source_value":"partner-1", "condition_source_value":"I10", "condition_start_date":"2024-01-01" }])
    assert storage.write_omop_staging(clean, "synthetic partner") == 1
    sql, rows = statements[0]
    assert "raw.partner_omop_staging" in sql
    assert rows == [("partner-1", "I10", "2024-01-01", "synthetic partner")]

