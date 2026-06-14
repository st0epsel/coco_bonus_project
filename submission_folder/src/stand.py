from pathlib import Path
from typing import Tuple

import pandas as pd


def run_initial_pos_sweep(
    x_steps: int = 7,
    p_steps: int = 7,
    perturbation_key: str = "main_thruster_range",
    perturbation_range: Tuple[float, float] = (0.5, 1.0),
    x_range: Tuple[float, float] = (0.0, 2.5),
    trials: int = 1,
    out_dir: str | None = None,
    show_plot: bool = False,
):
    """Run the initial-position vs perturbation sweep and save results.

    Saves:
      - results CSV to `<out_dir>/initial_pos_shape.csv`
      - figure to `<out_dir>/initial_pos_shape.png`

    Returns (fig, df)
    """
    from submission_folder.src.plot import find_initial_pos_shape_for_malfunction

    if out_dir is None:
        out_dir = Path(__file__).parent.parent / "results" / "initial_pos_shape"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, df = find_initial_pos_shape_for_malfunction(
        perturbation_key=perturbation_key,
        perturbation_range=perturbation_range,
        x_range=x_range,
        x_steps=x_steps,
        p_steps=p_steps,
        trials=trials,
        save_dir=str(out_dir),
        show_plot=show_plot,
    )

    # ensure dataframe saved
    csv_path = out_dir / "initial_pos_shape.csv"
    try:
        df.to_csv(csv_path, index=False)
    except Exception as e:
        print(f"Warning: failed to save results CSV: {e}")

    print(f"Saved figure and data to: {out_dir}")
    succ_count = int(df["success"].sum()) if "success" in df.columns else 0
    total = len(df)
    print(f"Summary: {succ_count}/{total} successful landings in grid sweep")
    return fig, df


if __name__ == "__main__":
    # Quick default run for convenience
    run_initial_pos_sweep(x_steps=7, p_steps=7, trials=1, show_plot=False)
