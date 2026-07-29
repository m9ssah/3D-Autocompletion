import csv


class ResultRecorder:
    """
    accumulates result rows and writes them to csv
    each row can carry arbitrary fields -- header is the union of all keys seen

    params (add):
    - **fields: arbitrary keyword fields for one result row, e.g.
      shape_id="monitor_1", method="pca", mask_type="halfspace", iou=0.8,
      sdf_error=0.03, solve_time_s=0.002, lam=1.0
    """

    def __init__(self):
        self.rows = []

    def add(self, **fields):
        self.rows.append(fields)

    def save(self, path):
        if not self.rows:
            raise ValueError("no rows to save")

        fieldnames = []
        for row in self.rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    recorder = ResultRecorder()
    recorder.add(shape_id="monitor_1", method="pca", mask_type="halfspace", iou=0.81, sdf_error=0.031, solve_time_s=0.002)
    recorder.add(shape_id="monitor_1", method="ae", mask_type="halfspace", iou=0.86, sdf_error=0.024, solve_time_s=0.41)
    recorder.add(shape_id="monitor_2", method="pca", mask_type="grid", iou=0.74, sdf_error=0.045, solve_time_s=0.002, lam=0.5)

    out_path = Path(tempfile.mkdtemp()) / "dummy_results.csv"
    recorder.save(out_path)

    print(f"saved to {out_path}")
    with open(out_path) as f:
        print(f.read())
