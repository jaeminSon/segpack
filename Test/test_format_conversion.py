import shutil
from glob import glob
from pathlib import Path

import pytest
import json
import numpy as np
from PIL import Image

from segpack.convert import convert_format


class TestFormatConversion:

    @pytest.fixture
    def remove_output(self):
        yield  # test

        # after test
        for p in glob("Test/output/*"):
            shutil.rmtree(p, ignore_errors=True)

    def test_labelmeFile2segmapFile(self, remove_output):
        convert_format("Test/samples/labelme", "Test/output/labelme2segmap", "labelme", "segmap",
                       {"__ignore__": -1, "_background_": 0, "1": 1, "2": 2, "3": 3})
        assert len(glob("Test/output/labelme2segmap/*")) > 0
        for p in Path("Test/output/labelme2segmap/").iterdir():
            assert len(np.array(Image.open(p)).shape) == 2  # seg map in format of (h, w)

    def test_segmapFile2labelmeFile(self, remove_output):
        convert_format("Test/samples/img_seg_pairs", "Test/output/segmap2labelme", "segmap", "labelme")
        assert len(glob("Test/output/segmap2labelme/*")) > 0
        for p in Path("Test/output/segmap2labelme/").iterdir():
            with open(p, "r") as f:
                json.load(f)
