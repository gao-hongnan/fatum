from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from .agents.cove import CoVeCandidate, CoVeOrchestrator
from .config.settings import get_settings
from .types import BatchVerificationResult, UserQuery, VerificationResult


async def run_verification(
    user_query: UserQuery,
    env_file: str | None = None,
    yaml_file: str | None = None,
) -> CoVeCandidate:
    """Run verification using CoVe (Chain-of-Verification)."""
    settings = get_settings(env_file=env_file, yaml_file=yaml_file)
    cove_config = settings.cove

    orchestrator = CoVeOrchestrator(config=cove_config)
    result = await orchestrator.aexecute(user_query)

    return result


async def run_batch_verification(
    user_queries: list[UserQuery],
    env_file: str | None = None,
    yaml_file: str | None = None,
) -> BatchVerificationResult:
    """Run verification on multiple user queries concurrently."""
    tasks = [
        run_verification(user_query=user_query, env_file=env_file, yaml_file=yaml_file) for user_query in user_queries
    ]
    results = await asyncio.gather(*tasks)

    verification_results = [
        VerificationResult(user_query=user_query, result=result)
        for user_query, result in zip(user_queries, results, strict=False)
    ]

    aligned_count = sum(1 for r in results if r.is_aligned)
    total_confidence = sum(r.confidence for r in results)

    return BatchVerificationResult(
        results=verification_results,
        total_questions=len(user_queries),
        aligned_count=aligned_count,
        average_confidence=total_confidence / len(results) if results else 0.0,
    )


def load_questions(file_path: str | Path) -> list[UserQuery]:
    file_path = Path(file_path)
    if file_path.suffix != ".json":
        raise ValueError(f"Unsupported file format: {file_path.suffix}. Use JSON.")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        user_queries = [UserQuery(**item) for item in data]

    return user_queries


async def main() -> None:
    """Main async function to run verifications."""
    parser = argparse.ArgumentParser(description="Run verification on text pairs")
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        help="Input file path (JSON) containing questions",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output file path to save batch results (JSON format)",
    )
    parser.add_argument(
        "--env-file",
        type=str,
        default="playground/critic/.env",
        help="Path to .env file (default: playground/critic/.env)",
    )
    parser.add_argument(
        "--yaml-file",
        type=str,
        default="playground/critic/config/config.yaml",
        help="Path to YAML config file (default: playground/critic/config/config.yaml)",
    )

    args = parser.parse_args()

    if args.input:
        print(f"Loading questions from {args.input}...")
        user_queries = load_questions(args.input)
        print(f"Loaded {len(user_queries)} questions")

        print("\n--- Running Batch CoVe Verification ---")
        start = time.perf_counter()
        batch_result = await run_batch_verification(
            user_queries,
            env_file=args.env_file,
            yaml_file=args.yaml_file,
        )
        end = time.perf_counter()
        print(f"Time taken: {end - start:.2f} seconds")
        print("\nBatch Results:")
        print(f"Total questions: {batch_result.total_questions}")
        print(f"Aligned answers: {batch_result.aligned_count}")
        print(f"Success rate: {batch_result.success_rate:.2%}")
        print(f"Average confidence: {batch_result.average_confidence:.2f}")

        if args.output:
            output_path = Path(args.output)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(batch_result.model_dump(), f, indent=4, ensure_ascii=False)
            print(f"\nResults saved to {output_path}")
    else:
        print("--- Running CoVe Verification ---")
        await run_verification(
            user_query=UserQuery(
                question="Who was the first woman to win two Nobel Prizes in different scientific fields?"
            ),
            env_file=args.env_file,
            yaml_file=args.yaml_file,
        )


if __name__ == "__main__":
    asyncio.run(main())

"""
uv run -m playground.critic.main \
    --env-file playground/critic/.env \
    --yaml-file playground/critic/config/config.yaml \
    --input playground/critic/questions.json \
    --output cove_results.json
"""
