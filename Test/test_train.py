import shutil
from glob import glob
from pathlib import Path

import pytest


from segpack import train


PATH_AIRSZOO_INHOUSE = ["/vision3/airszoo"]


class TestTrain:
    @pytest.fixture
    def set_inhouse_path_setting(self):
        # before test
        from airszoo import set_paths
        set_paths(PATH_AIRSZOO_INHOUSE, makedirs=False)

        yield  # test

        # after test
        shutil.rmtree("results")

    @pytest.mark.skipif(any([not Path(p).exists() for p in PATH_AIRSZOO_INHOUSE]), reason="Directory not found for inhouse setting.")
    def test_train_scratch(self, set_inhouse_path_setting):
        train("bodysealer_train", "bodysealer_val", "bodysealer_scratch", "bodysealer")
        assert len(glob("results/checkpoint/*")) == 1
        assert len(glob("results/tensorboard/*")) == 1

    @pytest.mark.skipif(any([not Path(p).exists() for p in PATH_AIRSZOO_INHOUSE]), reason="Directory not found for inhouse setting.")
    def test_train_continual(self, set_inhouse_path_setting):
        train("bodysealer_train", "bodysealer_val", "bodysealer", "bodysealer")
        assert len(glob("results/checkpoint/*")) == 1
        assert len(glob("results/tensorboard/*")) == 1