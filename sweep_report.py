"""
make_parallel_coordinates_report.py -- Q3: generate the actual Wandb
Parallel Coordinates CHART (not just log the metrics that feed it).

wandb.log() in your sweep cell only writes per-run scalars into each run's
history -- it does not create the parallel-coordinates PANEL itself. That
panel is a piece of a Workspace or Report, and has to be created explicitly,
either by hand (Workspace -> Add Panels -> Parallel coordinates) or via
W&B's Reports API, which is what this does.

Run this AFTER your sweep_configs loop has finished and every run
(w{weight_bits}_a{activation_bits}) is visible in your wandb project.

W&B's Reports API has moved packages over time (older: `wandb.apis.reports`,
requiring `wandb.require('report-editing')`; newer: the standalone
`wandb-workspaces` package, `wandb_workspaces.reports.v2`). This tries the
newer one first and falls back to the older one, since I can't verify
which is installed/current in your environment from here. If BOTH of these
fail (API surface shifted again), the guaranteed fallback is 30 seconds by
hand: open your project -> Workspace -> "Add Panels" -> "Parallel
coordinates" -> pick the same columns listed below -> screenshot it for
the PDF.
"""

ENTITY = "id25s001-iit-madras-foundation"   # from your run URLs
PROJECT = "cs6886-assignment2"

# must match exactly what you wandb.log() in the sweep loop
PC_COLUMNS = [
    "weight_quant_bits",
    "activation_quant_bits",
    "compression_ratio",
    "model_size_mb",
    "quantized_acc",
]


def _build_with_wandb_workspaces():
    import wandb_workspaces.reports.v2 as wr

    report = wr.Report(
        entity=ENTITY,
        project=PROJECT,
        title="Q3 -- Quantization Sweep: Parallel Coordinates",
        description=(
            "Weight/activation bit-width sweep vs. compression ratio, "
            "model size, and accuracy."
        ),
    )
    report.blocks = [
        wr.PanelGrid(
            panels=[
                wr.ParallelCoordinatesPlot(
                    columns=[wr.reports.PCColumn(c) for c in PC_COLUMNS],
                )
            ],
            runsets=[wr.Runset(entity=ENTITY, project=PROJECT)],
        )
    ]
    report.save()
    return report


def _build_with_legacy_api():
    import wandb
    wandb.require("report-editing")
    import wandb.apis.reports as wr

    report = wr.Report(
        entity=ENTITY,
        project=PROJECT,
        title="Q3 -- Quantization Sweep: Parallel Coordinates",
        blocks=[
            wr.PanelGrid(
                panels=[
                    wr.ParallelCoordinatesPlot(
                        columns=[wr.reports.PCColumn(c) for c in PC_COLUMNS],
                    )
                ],
                runsets=[wr.RunSet(entity=ENTITY, project=PROJECT)],
            ),
        ],
    )
    report.save()
    return report


if __name__ == "__main__":
    report = None
    errors = []
    for builder in (_build_with_wandb_workspaces, _build_with_legacy_api):
        try:
            report = builder()
            break
        except Exception as e:  # noqa: BLE001 -- deliberately broad, see note below
            errors.append(f"{builder.__name__}: {type(e).__name__}: {e}")

    if report is None:
        raise RuntimeError(
            "Both the wandb_workspaces and legacy wandb.apis.reports paths "
            "failed to build the report -- the API has likely moved again. "
            "Errors seen:\n  " + "\n  ".join(errors) +
            "\n\nFallback: open your project in the browser, go to the "
            "Workspace tab, click 'Add Panels', choose 'Parallel "
            f"coordinates', and add these columns manually: {PC_COLUMNS}"
        )

    print(f"Report saved: {report.url}")
    print(
        "Open that URL, confirm the parallel-coordinates panel renders with "
        "all your sweep runs, then screenshot/export it as an image for the "
        "assignment PDF -- a live report link isn't itself embeddable in a PDF."
    )
