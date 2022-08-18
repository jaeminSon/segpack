import os
import shutil
from pathlib import Path

import pytest

import airszoo


class TestUpoad:
    
    @pytest.fixture
    def set_path_setting_at_cwd(self):
        # before test
        from airszoo import set_paths
        set_paths(["."], makedirs=True)

        yield  # test

        # after test
        shutil.rmtree(Path.cwd() / "recipe")
        shutil.rmtree(Path.cwd() / "preprocess")
        shutil.rmtree(Path.cwd() / "augment")
        shutil.rmtree(Path.cwd() / "hyperparam")
        shutil.rmtree(Path.cwd() / "architecture")
        shutil.rmtree(Path.cwd() / "checkpoint")
        shutil.rmtree(Path.cwd() / "data")
        # remove path.json if exists
        p = Path.cwd() / "airszoo" / "path.json"
        if p.exists():
            os.remove(p)

    @pytest.mark.skipif(not Path("/vision3/airszoo/checkpoint/bodysealer.pth").exists(), reason="Directory not found for inhouse setting.")
    def test_upload(self, set_path_setting_at_cwd):
        airszoo.upload_trained_network(recipe_path="Test/samples/recipe_upload.json",
                                    checkpoint_path="/vision3/airszoo/checkpoint/bodysealer.pth", 
                                    dest_filename="bodysealer_test", 
                                    architecture_path="Test/samples/deeplab_v3plus.py")
        assert (Path.cwd() / "recipe" / "bodysealer_test.json").exists()
        assert (Path.cwd() / "architecture" / "deeplab_bodysealer.py").exists()
        assert (Path.cwd() / "checkpoint" / "bodysealer_test.pth").exists()
