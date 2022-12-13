import shutil
from glob import glob
from pathlib import Path

import pytest


from segpack import train


class TestTrain:
    @pytest.fixture
    def set_inhouse_path_setting(self):
        yield  # test

        # after test
        shutil.rmtree("results")

    def test_train_scratch(self, set_inhouse_path_setting):
        train("bodysealer_train", "bodysealer_val", "bodysealer_scratch", "bodysealer")
        assert len(glob("results/checkpoint/*")) == 1
        assert len(glob("results/tensorboard/*")) == 1

    def test_train_continual(self, set_inhouse_path_setting):
        train("bodysealer_train", "bodysealer_val", "bodysealer", "bodysealer")
        assert len(glob("results/checkpoint/*")) == 1
        assert len(glob("results/tensorboard/*")) == 1