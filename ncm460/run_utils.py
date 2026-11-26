# bookkeeping of the runs: run folders and the registry of finished stages (the helpers themselves are in pdgraph)
import config as cfg
from pdgraph.metrics import top1_and_macro_f1  # noqa: F401  (re-exported for the scripts)
from pdgraph.runs import RunRegistry, create_run_dir, make_writer, settings_snapshot  # noqa: F401
from pdgraph.utils import WarmupCosineSchedule, build_optimizer  # noqa: F401

# a tagged run takes the stages listed in RUN_TAG_INHERITS from the main runs
_registry = RunRegistry(cfg.RUNS_DIR, fallback_dir=cfg.BASE_RUNS_DIR if cfg.BASE_RUNS_DIR != cfg.RUNS_DIR else None, fallback_stages=cfg.RUN_TAG_INHERITS)


def new_run_dir(stage_name, source_file, seed=None):
    return create_run_dir(cfg.RUNS_DIR, stage_name, source_file, settings_snapshot(cfg), seed)


def register_run(stage, run_dir, seed=None):
    _registry.register(stage, run_dir, seed)


def registered_run(stage, seed=None):
    return _registry.find(stage, seed)
