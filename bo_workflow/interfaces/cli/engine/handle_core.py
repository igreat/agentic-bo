from __future__ import annotations

import argparse

from ....engine import BOEngine
from .common import json_print, parse_json_object, parse_observation_records


def handle_core(args: argparse.Namespace, engine: BOEngine) -> int | None:
    if args.command == "init":
        intent_payload = None
        if args.intent_json is not None:
            intent_payload = parse_json_object(args.intent_json)
        payload = engine.init_run(
            dataset_path=args.dataset,
            target_column=args.target,
            objective=args.objective,
            default_engine=args.engine,
            run_id=args.run_id,
            seed=args.seed,
            num_initial_random_samples=args.init_random,
            default_batch_size=args.batch_size,
            max_categories=args.max_categories,
            intent=intent_payload,
            verbose=args.verbose,
        )
        json_print(payload)
        return 0

    if args.command == "suggest":
        payload = engine.suggest(
            args.run_id,
            batch_size=args.batch_size,
            verbose=args.verbose,
        )
        json_print(payload)
        return 0

    if args.command == "observe":
        observations = parse_observation_records(args.data)
        payload = engine.observe(args.run_id, observations, verbose=args.verbose)
        json_print(payload)
        return 0

    if args.command == "status":
        payload = engine.status(args.run_id)
        json_print(payload)
        return 0

    if args.command == "report":
        payload = engine.report(args.run_id, verbose=args.verbose)
        json_print(payload)
        return 0

    return None
