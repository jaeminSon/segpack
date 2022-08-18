import shutil
from glob import glob
from pathlib import Path

import pytest

from segpack.deploy import generate_jit, Deployer

PATH_AIRSZOO_INHOUSE = ["/vision3/airszoo"]

class TestJit:
    
    @pytest.fixture
    def remove_output(self):
        yield  # test

        # after test
        for p in glob("Test/output/*"):
            shutil.rmtree(p, ignore_errors=True)
    
    @pytest.mark.skipif(any([not Path(p).exists() for p in PATH_AIRSZOO_INHOUSE]), reason="Directory not found for inhouse setting.")
    def test_generate_jit(self):
        generate_jit("bodysealer", 2049, 2449, "Test/output/jit")
        
    def test_inference_jit(self, remove_output):
        deployer = Deployer("Test/output/jit/network_jit.pt", (2049,2449), (2048,2448), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        result = deployer.inference("Test/samples/img_seg_pairs/image/1.jpg")
        assert result.shape == (2048,2448)
