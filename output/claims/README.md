# Claims

One JSON file per reported claim. Create it with `ClaimStore`; do not hand-copy
numbers into a paper or report.

```python
from dlef.claim import Claim, ClaimStore

ClaimStore("output/claims").add(
    Claim(
        claim_id="main-result",
        text="Our method improves the main metric",
        eval_ids=("eval-id-from-a-run",),
        metric="mae",
    )
)
```

Then:

```bash
python -m dlef.cli report --runs output/runs --claims output/claims --out output/reports/report.md
```

Rules:

- A claim must reference at least one eval id.
- A test claim should cite the eval record from a test-split run.
- `test_access > 1` is shown as `WARN`; do not hide it.
- The report resolves eval → checkpoint → run; missing eval ids show `MISSING`.
