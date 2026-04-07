"""
Tests for the Optimizer wrapper.
"""
import pytest
import torch
import torch.nn as nn
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pathogenfinder2.dl.utils.optimizer import Optimizer


# ---------------------------------------------------------------------------
# Fixture: a tiny network to optimise
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_network():
    return nn.Linear(4, 1)


# ---------------------------------------------------------------------------
# Optimizer construction
# ---------------------------------------------------------------------------

class TestOptimizerInit:

    def test_adam_sets_amsgrad(self, tiny_network):
        opt = Optimizer(tiny_network, torch.optim.Adam, 1e-3, 1e-4, amsgrad=True)
        assert opt.optimizer.param_groups[0]["amsgrad"] is True

    def test_adam_no_amsgrad(self, tiny_network):
        opt = Optimizer(tiny_network, torch.optim.Adam, 1e-3, 1e-4, amsgrad=False)
        assert opt.optimizer.param_groups[0]["amsgrad"] is False

    def test_adamw_instantiates(self, tiny_network):
        opt = Optimizer(tiny_network, torch.optim.AdamW, 1e-3, 1e-4, amsgrad=False)
        assert isinstance(opt.optimizer, torch.optim.AdamW)

    def test_nadam_instantiates(self, tiny_network):
        opt = Optimizer(tiny_network, torch.optim.NAdam, 1e-3, 1e-4, amsgrad=False)
        assert isinstance(opt.optimizer, torch.optim.NAdam)

    def test_learning_rate_stored(self, tiny_network):
        opt = Optimizer(tiny_network, torch.optim.Adam, 5e-4, 1e-4, amsgrad=False)
        assert opt.learning_rate == 5e-4

    def test_lr_set_in_optimizer(self, tiny_network):
        opt = Optimizer(tiny_network, torch.optim.Adam, 3e-3, 1e-4, amsgrad=False)
        assert opt.optimizer.param_groups[0]["lr"] == pytest.approx(3e-3)

    def test_weight_decay_set(self, tiny_network):
        opt = Optimizer(tiny_network, torch.optim.Adam, 1e-3, 5e-5, amsgrad=False)
        assert opt.optimizer.param_groups[0]["weight_decay"] == pytest.approx(5e-5)

    def test_scheduler_initially_none(self, tiny_network):
        opt = Optimizer(tiny_network, torch.optim.Adam, 1e-3, 1e-4, amsgrad=False)
        assert opt.lr_scheduler is None

    def test_warmup_initially_none(self, tiny_network):
        opt = Optimizer(tiny_network, torch.optim.Adam, 1e-3, 1e-4, amsgrad=False)
        assert opt.warmup is None


# ---------------------------------------------------------------------------
# Schedulers
# ---------------------------------------------------------------------------

class TestSetScheduler:

    def _opt(self, net):
        return Optimizer(net, torch.optim.Adam, 1e-3, 1e-4, amsgrad=False)

    def test_reduce_on_plateau(self, tiny_network):
        opt = self._opt(tiny_network)
        opt.set_scheduler("ReduceLROnPlateau")
        assert isinstance(opt.lr_scheduler,
                          torch.optim.lr_scheduler.ReduceLROnPlateau)

    def test_multistep_lr(self, tiny_network):
        opt = self._opt(tiny_network)
        opt.set_scheduler("MultiStepLR")
        assert isinstance(opt.lr_scheduler,
                          torch.optim.lr_scheduler.MultiStepLR)

    def test_onecycle_lr(self, tiny_network):
        opt = self._opt(tiny_network)
        opt.set_scheduler("OneCycleLR", end_lr=1e4, steps=10, epochs=5)
        assert isinstance(opt.lr_scheduler,
                          torch.optim.lr_scheduler.OneCycleLR)

    def test_invalid_scheduler_raises(self, tiny_network):
        opt = self._opt(tiny_network)
        with pytest.raises(ValueError):
            opt.set_scheduler("NonExistentScheduler")


# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------

class TestSetWarmup:

    def test_warmup_set(self, tiny_network):
        opt = Optimizer(tiny_network, torch.optim.Adam, 1e-3, 1e-4, amsgrad=False)
        opt.set_warmup(warmup_period=100)
        assert opt.warmup is not None
        assert opt.warmup["period"] == 100

    def test_warmup_scheduler_created(self, tiny_network):
        import pytorch_warmup as warmup
        opt = Optimizer(tiny_network, torch.optim.Adam, 1e-3, 1e-4, amsgrad=False)
        opt.set_warmup(50)
        assert isinstance(opt.warmup["scheduler"], warmup.LinearWarmup)


# ---------------------------------------------------------------------------
# scheduler_step / update_scheduler
# ---------------------------------------------------------------------------

class TestSchedulerStep:

    def test_plateau_step_with_value(self, tiny_network):
        opt = Optimizer(tiny_network, torch.optim.Adam, 1e-3, 1e-4, amsgrad=False)
        opt.set_scheduler("ReduceLROnPlateau")
        opt.scheduler_step(value=0.5)  # should not raise

    def test_multistep_step_no_value(self, tiny_network):
        opt = Optimizer(tiny_network, torch.optim.Adam, 1e-3, 1e-4, amsgrad=False)
        opt.set_scheduler("MultiStepLR")
        opt.scheduler_step()  # should not raise

    def test_update_scheduler_without_warmup(self, tiny_network):
        opt = Optimizer(tiny_network, torch.optim.Adam, 1e-3, 1e-4, amsgrad=False)
        opt.set_scheduler("MultiStepLR")
        opt.update_scheduler()  # should not raise
