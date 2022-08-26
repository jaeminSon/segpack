# Segmentation for pseudolabeling and continal learning

### Install
``` 
pip install git+https://github.hmckmc.co.kr/airlab/segpack
(github ID, password 입력 필요)
```

### Usage
``` 
# training
>>> from segpack import train
>>> train("train_data_name", "val_data_name", "pretrained_network_name", "config_name")

# upload trained model
>>> from airszoo import upload_trained_network
>>> upload_trained_network("new_recipe.json",  "checkpoint.pth", dest_filename="new_recipe_name")

# inference directory structure
root
└── inference
    └── image_file1.jpeg
    └── image_file2.jpeg
    └── ...

# generate png (segmentation map)
>>> from segpack.convert import generate_pseudolabels
>>> generate_pseudolabels("path/to/root", "pretrained_network_name", "segmap", "path/to/output/dir", "segmentation_resize")

# generate labelme readable json
>>> from segpack.convert import generate_pseudolabels
>>> generate_pseudolabels("path/to/root", "pretrained_network_name", "labelme", "path/to/output/dir", "segmentation_resize")

# convert to torchscript model (for faster inference)
>>> from segpack.deploy import generate_jit
>>> generate_jit("bodysealer_test", 2049, 2449, "output/jit")

# run torchscript model (copy codes of Deployer for actual deployment)
>>> from segpack.deploy import Deployer
>>> deployer = Deployer("output/jit/network_jit.pt", (2049,2449), (2048,2448), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
>>> result = deployer.inference("samples/img_seg_pairs/image/1.jpg")

# convert format (png -> labelme)
>>> from segpack.convert import convert_format
>>> convert_format("samples/img_seg_pairs", "path/to/output/dir", "segmap", "labelme")

# convert format (labelme -> png)
>>> from segpack.convert import convert_format
>>> convert_format("samples/labelme", "path/to/output/dir", "labelme", "segmap", {"__ignore__": -1, "_background_": 0, "1": 1, "2": 2, "3": 3})
```

### Examples (original image - png - labelme)
<img width="300" alt="image" src="https://user-images.githubusercontent.com/8290383/186805557-2225fb03-2674-49ba-a473-ff581ac8fce7.jpeg"> <img width="300" alt="image" src="https://user-images.githubusercontent.com/8290383/186805606-616c6698-1f2d-440d-a500-aa8641180fdc.png">
<img width="300" alt="스크린샷 2022-07-05 오후 3 29 42" src="https://user-images.githubusercontent.com/8290383/186805624-8b093902-d85a-4fe9-a300-d894dab1eea4.png">

### Segmentation data folder structure 
<img width="600" alt="스크린샷 2022-08-26 오전 11 39 45" src="https://user-images.githubusercontent.com/8290383/186805750-8f5bf975-48ba-43b8-90a5-6a09fe0873f4.png">

### Inference data folder structure 
<img width="600" alt="스크린샷 2022-08-26 오전 11 39 49" src="https://user-images.githubusercontent.com/8290383/186805790-97571085-79ad-4557-997f-7a68b2a7b302.png">

