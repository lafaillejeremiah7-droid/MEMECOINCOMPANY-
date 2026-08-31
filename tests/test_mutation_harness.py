from subprocess import CompletedProcess
from unittest.mock import patch

import pytest

from scripts import mutation_check


def test_failed_git_status_never_reports_clean():
    with patch.object(mutation_check, "_run", return_value=CompletedProcess([], 1, "", "denied")), \
            pytest.raises(RuntimeError, match="Cannot verify working tree"):
        mutation_check._dirty(["memescanner/database.py"])


def example_mutation(tmp_path):
    path = tmp_path / "example.py"
    original = "# Unicode — and CRLF\r\nLIMIT = 99\r\n".encode("utf-8")
    path.write_bytes(original)
    mutation = mutation_check.Mutation("example", str(path), "LIMIT = 99", "LIMIT = 0", "unused", "test")
    return path, original, mutation


def test_isolated_mutation_preserves_original_bytes(tmp_path):
    path, original, mutation = example_mutation(tmp_path)
    with mutation_check._isolated_case({"example.py": original}) as workspace:
        isolated = mutation_check.replace(mutation, path=str(workspace / "example.py"))
        assert mutation_check._apply(isolated)
        assert (workspace / "example.py").read_bytes() != original
        assert path.read_bytes() == original
    assert not workspace.exists()
    assert path.read_bytes() == original


def test_failed_case_cannot_modify_original_checkout(tmp_path):
    path, original, mutation = example_mutation(tmp_path)
    with pytest.raises(RuntimeError, match="test failure"), \
            mutation_check._isolated_case({"example.py": original}) as workspace:
        isolated = mutation_check.replace(mutation, path=str(workspace / "example.py"))
        assert mutation_check._apply(isolated)
        raise RuntimeError("test failure")
    assert path.read_bytes() == original
    assert not workspace.exists()


def test_missing_mutation_pattern_leaves_source_untouched(tmp_path):
    path, original, mutation = example_mutation(tmp_path)
    missing = mutation_check.Mutation("absent", str(path), "ABSENT", "new", "unused", "test")
    assert not mutation_check._apply(missing)
    assert path.read_bytes() == original


def test_each_case_starts_from_fresh_source_without_bytecode(tmp_path):
    path, original, mutation = example_mutation(tmp_path)
    with mutation_check._isolated_case({"example.py": original}) as first:
        assert mutation_check._apply(mutation_check.replace(mutation, path=str(first / "example.py")))
        (first / "__pycache__").mkdir()
        (first / "__pycache__" / "stale.pyc").write_bytes(b"stale")
    with mutation_check._isolated_case({"example.py": original}) as second:
        assert (second / "example.py").read_bytes() == original
        assert not (second / "__pycache__").exists()
    assert path.read_bytes() == original
