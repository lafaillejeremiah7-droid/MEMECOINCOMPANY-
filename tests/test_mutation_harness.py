from subprocess import CompletedProcess
from unittest.mock import patch

import pytest

from scripts import mutation_check


def test_failed_git_status_never_reports_clean():
    with patch.object(mutation_check, "_run", return_value=CompletedProcess([], 1, "", "denied")), \
            pytest.raises(RuntimeError, match="Cannot verify working tree"):
        mutation_check._dirty(["memescanner/database.py"])


@pytest.mark.parametrize("results", [[1], [0, 1]])
def test_failed_restore_never_claims_mutation_reverted(results):
    with patch.object(mutation_check, "_run", side_effect=[CompletedProcess([], code, "", "") for code in results]), \
            pytest.raises(RuntimeError, match="restore"):
        mutation_check._revert(mutation_check.MUTATIONS[0])
