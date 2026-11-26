import json
import shutil
from datetime import datetime
from pathlib import Path


def settings_snapshot(config_module):
    # the upper-case settings of a config module as a json-serialisable dict
    simple = (int, float, str, bool, list, tuple, dict, type(None))
    return {k: (v if isinstance(v, simple) else str(v)) for k, v in vars(config_module).items() if k.isupper()}


def create_run_dir(runs_dir, stage_name, source_file, settings, seed=None):
    # runs_dir/<stage_name>[_seed<seed>]_<timestamp> with a copy of the script and a snapshot of the settings
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(runs_dir) / (f"{stage_name}_seed{seed}_{stamp}" if seed is not None else f"{stage_name}_{stamp}")
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(source_file, run_dir)
    (run_dir / "settings_snapshot.json").write_text(json.dumps(settings, indent=2, default=str), encoding="utf-8")
    return run_dir


class RunRegistry:
    # runs_dir/registry.json records which run folder holds which stage (and seed), so that the next stage finds its checkpoint without typing.
    # fallback_stages are looked up in fallback_dir when they are not registered in runs_dir (a tagged run that reuses the stage of the main run).
    def __init__(self, runs_dir, fallback_dir=None, fallback_stages=()):
        self.runs_dir = Path(runs_dir)
        self.fallback_dir = Path(fallback_dir) if fallback_dir is not None else None
        self.fallback_stages = tuple(fallback_stages)

    @staticmethod
    def _read(folder):
        path = Path(folder) / "registry.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def register(self, stage, run_dir, seed=None):
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        registry = self._read(self.runs_dir)
        if seed is None:
            registry[stage] = run_dir.name
        else:
            registry.setdefault(stage, {})[str(seed)] = run_dir.name
        (self.runs_dir / "registry.json").write_text(json.dumps(registry, indent=2), encoding="utf-8")

    def find(self, stage, seed=None):
        # folder of a registered run, or None
        entry, folder = self._read(self.runs_dir).get(stage), self.runs_dir
        if entry is None and stage in self.fallback_stages and self.fallback_dir is not None:
            entry, folder = self._read(self.fallback_dir).get(stage), self.fallback_dir
        if seed is not None and isinstance(entry, dict):
            entry = entry.get(str(seed))
        return folder / entry if entry else None


def make_writer(run_dir):
    # TensorBoard writer when tensorboard is installed, otherwise a silent stand-in (training does not depend on it)
    try:
        from torch.utils.tensorboard import SummaryWriter
        return SummaryWriter(str(run_dir))
    except Exception:
        class _SilentWriter:
            def add_scalar(self, *args, **kwargs):
                pass

            def close(self):
                pass
        return _SilentWriter()
