import shutil
from pathlib import Path
from glob import glob

import pytest
import json
import numpy as np
from PIL import Image

from segpack.convert import generate_pseudolabels


class TestFormatConversion:

    @pytest.fixture
    def remove_output(self):
        yield  # test

        # after test
        shutil.rmtree("Test/output")

    def test_generate_segmap(self, remove_output):
        generate_pseudolabels("Test/val", "bodysealer", "segmap", "Test/output/segmap", "segmentation_resize")
        assert len(glob("Test/output/segmap/*")) > 0
        for p in Path("Test/output/segmap/").iterdir():
            assert len(np.array(Image.open(p)).shape) == 2  # seg map in format of (h, w)

    def test_generate_labelme(self, remove_output):
        generate_pseudolabels("Test/val", "bodysealer", "labelme", "Test/output/labelme", "segmentation_resize")
        assert len(glob("Test/output/labelme/*")) > 0
        for p in Path("Test/output/labelme/").iterdir():
            with open(p, "r") as f:
                json.load(f)
