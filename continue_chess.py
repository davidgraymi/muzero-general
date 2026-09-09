import pathlib
import sys

import ray
import torch

from muzero import MuZero


def main():
    target_step = int(sys.argv[1]) if len(sys.argv) > 1 else 51000

    runs = sorted(
        (
            path
            for path in pathlib.Path("results/chess").glob("*/")
            if (path / "model.checkpoint").is_file()
            and (path / "replay_buffer.pkl").is_file()
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    if not runs:
        raise FileNotFoundError(
            "No complete chess training runs found"
        )

    run_dir = runs[0]
    checkpoint_path = run_dir / "model.checkpoint"
    replay_buffer_path = run_dir / "replay_buffer.pkl"
    best_model_path = run_dir / "best_model.checkpoint"

    if not best_model_path.is_file():
        raise FileNotFoundError(
            f"Best model not found: {best_model_path}"
        )

    print(f"Continuing from: {run_dir}")
    print(f"Training checkpoint: {checkpoint_path}")
    print(f"Replay buffer: {replay_buffer_path}")
    print(f"Previous best model: {best_model_path}")
    print(f"Target training step: {target_step}")

    muzero = MuZero(
        "chess",
        config={
            "training_steps": target_step,
        },
    )

    muzero.load_model(
        checkpoint_path=checkpoint_path,
        replay_buffer_path=replay_buffer_path,
    )

    best_model = torch.load(
        best_model_path,
        map_location="cpu",
        weights_only=False,
    )

    if "weights" not in best_model:
        raise KeyError(
            f"{best_model_path} does not contain 'weights'"
        )

    muzero.previous_best_weights = best_model["weights"]
    muzero.previous_best_score = best_model.get("score", 0.5)
    muzero.previous_best_step = best_model.get(
        "training_step",
        0,
    )

    print(
        "Using previous best model from step "
        f"{muzero.previous_best_step} with score "
        f"{muzero.previous_best_score:.3f}"
    )

    muzero.train()


if __name__ == "__main__":
    try:
        main()
    finally:
        ray.shutdown()