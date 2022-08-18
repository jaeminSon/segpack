import shutil
from pathlib import Path
from glob import glob

import pytest
import json
import numpy as np
from PIL import Image

from segpack.convert import generate_pseudolabels

PATH_AIRSZOO_INHOUSE = ["/vision3/airszoo"]


class TestFormatConversion:

    @pytest.fixture
    def set_inhouse_path_setting(self):
        # before test
        from airszoo import set_paths
        set_paths(PATH_AIRSZOO_INHOUSE, makedirs=False)

        yield  # test

        # after test
        for p in glob("Test/output/*"):
            shutil.rmtree(p, ignore_errors=True)

    @pytest.mark.skipif(any([not Path(p).exists() for p in PATH_AIRSZOO_INHOUSE]), reason="Directory not found for inhouse setting.")
    def test_generate_segmap(self, set_inhouse_path_setting):
        generate_pseudolabels("Test/val", "bodysealer", "segmap", "Test/output/segmap", "segmentation_resize")
        assert len(glob("Test/output/segmap/*")) > 0
        for p in Path("Test/output/segmap/").iterdir():
            assert len(np.array(Image.open(p)).shape) == 2  # seg map in format of (h, w)

    @pytest.mark.skipif(any([not Path(p).exists() for p in PATH_AIRSZOO_INHOUSE]), reason="Directory not found for inhouse setting.")
    def test_generate_labelme(self, set_inhouse_path_setting):
        generate_pseudolabels("Test/val", "bodysealer", "labelme", "Test/output/labelme", "segmentation_resize")
        assert len(glob("Test/output/labelme/*")) > 0
        for p in Path("Test/output/labelme/").iterdir():
            with open(p, "r") as f:
                json.load(f)
