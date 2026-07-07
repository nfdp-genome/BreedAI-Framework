import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from breedai.runner.orchestrator import create_run


def test_dry_run_creates_run_folder_and_sbatch(tmp_path):
    result = create_run(
        repo_root=REPO_ROOT,
        phase=1,
        mode="default",
        species="cattle",
        goal="growth",
        input_type="vcf",
        run_id="test_dry_run",
        dry_run=True,
        submit=False,
        slurm=True,
    )
    run_dir = Path(result["run_dir"])
    assert run_dir.exists()
    assert (run_dir / "resolved_policy.yaml").exists()
    assert (run_dir / "plan.json").exists()
    assert (run_dir / "submit.sbatch").exists()
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["run_id"] == "test_dry_run"

