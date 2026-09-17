"""Lightning observer; never participates in the loss or modifies model outputs."""

import json
import os
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import lightning
from lightning import Callback

from amgm.utils import _to_json_safe
from .statistics import GateAccumulator, assignment_change


def write_manifest(path, manifest):
    Path(path).write_text(json.dumps(_to_json_safe(manifest), indent=2, default=str), encoding="utf-8")


class GatingDiagnostics(Callback):
    def __init__(self, output_dir, seed, config, train_subset, val_subset):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.phase = None
        self.pending = None
        self.handle = None
        self.fit_active = False
        self.seen_epochs = set()
        self.epoch_rows, self.batch_rows, self.change_rows = [], [], []
        self.previous = None
        dataset = train_subset.dataset
        self.val_indices = np.asarray(val_subset.indices)
        self.expected_keys = [f"{dataset.issue_ids[i]}|{dataset.test_dates[i]}" for i in self.val_indices]
        self.manifest = dict(
            schema_version=1, seed=seed, configuration=config,
            issue_ids=sorted(set(dataset.issue_ids) | set(getattr(dataset, "excluded_iids", []))),
            training_issue_ids=sorted(set(dataset.issue_ids)),
            training_excluded_issue_ids=list(getattr(dataset, "excluded_iids", [])),
            split_membership="split_membership.csv",
            runtime=dict(python=platform.python_version(), torch=torch.__version__,
                         lightning=lightning.__version__, numpy=np.__version__, platform=platform.platform()),
            training_history=True, experiment1_status="collecting",
            torch_compile_disable=os.environ.get("TORCH_COMPILE_DISABLE", ""),
            epochs=[], checkpoint=None,
        )
        rows = []
        for split, subset in [("train", train_subset), ("validation", val_subset)]:
            for i in subset.indices:
                rows.append(dict(dataset_index=i, split=split, issue_id=dataset.issue_ids[i],
                                 date=str(dataset.test_dates[i])))
        pd.DataFrame(rows).to_csv(self.output_dir / "split_membership.csv", index=False)
        self._save()

    def _save(self):
        write_manifest(self.output_dir / "manifest.json", self.manifest)
        for name, rows in [("training_batches", self.batch_rows), ("epoch_metrics", self.epoch_rows),
                           ("assignment_changes", self.change_rows)]:
            if rows:
                pd.DataFrame(rows).to_csv(self.output_dir / (name + ".csv"), index=False)

    def _observe(self, module, inputs, output):
        if self.phase is not None:
            # copy owns its CPU memory; no autograd graph or model tensor escapes.
            self.pending = output[2].detach().cpu().numpy().copy()

    def on_fit_start(self, trainer, pl_module):
        self.fit_active = True
        self.handle = pl_module.model.register_forward_hook(self._observe)

    def on_train_epoch_start(self, trainer, pl_module):
        self.train_stats = GateAccumulator()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.phase, self.pending = "train", None

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        p = self._take()
        self.train_stats.update(p)
        stats = GateAccumulator()
        stats.update(p)
        self.batch_rows.append(dict(epoch=trainer.current_epoch, batch=batch_idx,
                                    step=trainer.global_step, **stats.summary()))

    def _take(self):
        p = self.pending
        self.pending, self.phase = None, None
        if p is None:
            raise RuntimeError("Diagnostics did not observe the MoE forward output")
        return p

    def on_train_epoch_end(self, trainer, pl_module):
        self.epoch_rows.append(dict(epoch=trainer.current_epoch, split="train", **self.train_stats.summary()))
        self._save()

    def on_validation_epoch_start(self, trainer, pl_module):
        self.collect_val = self.fit_active and not trainer.sanity_checking and trainer.current_epoch not in self.seen_epochs
        self.val_parts, self.val_keys = [], []

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        if self.collect_val:
            self.phase, self.pending = "validation", None

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if self.collect_val:
            self.val_parts.append(self._take())
            self.val_keys.extend(f"{iid}|{date}" for iid, date in zip(batch.issue_ids, batch.test_dates))

    def on_validation_epoch_end(self, trainer, pl_module):
        if not self.collect_val:
            return
        if self.val_keys != self.expected_keys:
            raise ValueError("Validation order/cohort changed; cannot compare identical windows")
        p = np.concatenate(self.val_parts)
        stats = GateAccumulator()
        stats.update(p)
        epoch = trainer.current_epoch
        self.epoch_rows.append(dict(epoch=epoch, split="validation", **stats.summary()))
        np.savez_compressed(self.output_dir / f"validation_epoch_{epoch:03d}.npz",
                            probabilities=p, dataset_index=self.val_indices,
                            sample_key=np.asarray(self.val_keys))
        if self.previous is not None:
            self.change_rows.append(dict(epoch=epoch, **assignment_change(self.previous, p)))
        self.previous = p
        self.val_parts, self.val_keys = [], []
        self.seen_epochs.add(epoch)
        self.manifest["epochs"] = sorted(self.seen_epochs)
        self._save()

    def _remove_hook(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
        self.pending, self.phase = None, None
        self.fit_active = False

    def on_fit_end(self, trainer, pl_module):
        self._remove_hook()
        self.manifest["experiment1_status"] = "complete"
        self.manifest["checkpoint"] = trainer.checkpoint_callback.best_model_path if trainer.checkpoint_callback else None
        self._save()

    def on_exception(self, trainer, pl_module, exception):
        self._remove_hook()
        self.manifest["experiment1_status"] = "interrupted"
        self._save()

    def teardown(self, trainer, pl_module, stage):
        self._remove_hook()
