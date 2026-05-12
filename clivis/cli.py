"""Command-line entry points for CLiViS."""

import argparse
import json
from pathlib import Path


def build_parser():
    parser = argparse.ArgumentParser(
        prog="clivis",
        description="Run CLiViS video question-answering inference.",
    )
    parser.add_argument(
        "--video",
        required=True,
        help="Path to the input video file.",
    )
    parser.add_argument(
        "--question",
        required=True,
        help="Question to answer from the video.",
    )
    parser.add_argument(
        "--output-path",
        default="output_segments/output",
        help="Output prefix for generated video segments.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=1,
        help="Base video sampling FPS for VLM calls.",
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=360 * 480,
        help="Maximum video pixel budget passed to the VLM where supported.",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=15,
        help="Maximum iterative reasoning rounds.",
    )
    parser.add_argument(
        "--save-result",
        help="Optional JSON file path for saving the answer and run metadata.",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file does not exist: {video_path}")

    from clivis.pipeline.time_inference_od_neo4j import inference

    answer, metadata = inference(
        video_path=str(video_path),
        question=args.question,
        output_path=args.output_path,
        fps=args.fps,
        max_pixels=args.max_pixels,
        max_rounds=args.max_rounds,
    )

    result = {
        "answer": answer,
        "metadata": metadata,
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.save_result:
        save_path = Path(args.save_result)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
