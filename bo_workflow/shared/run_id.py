from datetime import UTC, datetime
import secrets

_RUN_ADJECTIVES = (
    "amber",
    "brisk",
    "crisp",
    "daring",
    "eager",
    "fuzzy",
    "gentle",
    "jolly",
    "lively",
    "nimble",
    "rapid",
    "steady",
    "sunny",
    "vivid",
)

_RUN_NOUNS = (
    "otter",
    "falcon",
    "heron",
    "lynx",
    "fox",
    "orca",
    "panda",
    "sparrow",
    "badger",
    "koala",
    "wolf",
    "tiger",
    "eagle",
    "whale",
)


def utc_now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


def generate_run_id() -> str:
    adjective = secrets.choice(_RUN_ADJECTIVES)
    noun = secrets.choice(_RUN_NOUNS)
    suffix = secrets.randbelow(10000)
    return f"{adjective}-{noun}-{suffix:04d}"
