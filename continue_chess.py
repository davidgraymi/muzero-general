import pathlib
import sys

import ray

from muzero import MuZero


def main():
    target_step = int(sys.argv[1]) if len(sys.argv) > 1 else 5000

    runs = sorted(
        pathlib.Path("results/chess").glob("*/"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    if not runs:
        raise FileNotFoundError("No chess training runs found")

    run_dir = runs[0]
    checkpoint = run_dir / "model.checkpoint"
    replay_buffer = run_dir / "replay_buffer.pkl"

    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    if not replay_buffer.is_file():
        raise FileNotFoundError(replay_buffer)

    print(f"Continuing from: {run_dir}")
    print(f"Target training step: {target_step}")

    muzero = MuZero(
        "chess",
        config={
            "training_steps": target_step,
        },
    )

    muzero.load_model(
        checkpoint_path=checkpoint,
        replay_buffer_path=replay_buffer,
    )

    muzero.train()


if __name__ == "__main__":
    try:
        main()
    finally:
        ray.shutdown()
