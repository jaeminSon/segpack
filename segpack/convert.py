import os
import json
import base64

from glob import glob
from pathlib import Path
from PIL import Image

import cv2
import numpy as np
import torch
from skimage.measure import label
from tqdm import tqdm
import labelme

import airszoo

from .utils import load_network, cudafy

NAME_LABELME = "labelme"
NAME_SEGMAP = "segmap"

EXT_LABELME = "json"
EXT_SEGMAP = "png"

__all__ = ['generate_pseudolabels', 'save_output', 'convert_format_from_ndarray', 'convert_format']


def generate_pseudolabels(data, pretrained_model: str, format: str, outdir: str, augment: str = None):
    assert not Path(outdir).exists(), "{} already exists.".format(outdir)

    network = load_network(pretrained_model)
    dataloader = airszoo.get_dataloader(data,
                                        preprocess=airszoo.get_preprocess_name_used_for_train(pretrained_model),
                                        augment=augment,
                                        **{"num_workers": 1,
                                            "pin_memory": True,
                                            "batch_size": 1,
                                            "shuffle": False})
    network.eval()
    tbar = tqdm(dataloader)
    for batch in tbar:
        image = cudafy(batch[0])[0] # cudafy returns list
        filepath = batch[1][0] # batch[1]==['path/to/something']
        with torch.no_grad():
            output = network(image)
        pred = output.data.cpu().numpy()
        pred = np.argmax(pred, axis=1)
        pred = np.squeeze(pred)

        converted_format = convert_format_from_ndarray(pred, format, filepath)

        save_output(converted_format, format, outdir, Path(filepath).name)


def save_output(object: object, object_format: str, outdir: str, filename: str):

    os.makedirs(outdir, exist_ok=True) 

    fid, _ = os.path.splitext(filename)
    if object_format == NAME_LABELME:
        extension = "."+EXT_LABELME
        path_out = Path(outdir) / (fid + extension)

        def convert(o):
            if isinstance(o, np.int64):
                return int(o)

        with open(path_out, 'w') as outfile:
            json.dump(object, outfile, default=convert)

    elif object_format == NAME_SEGMAP:
        extension = "."+EXT_SEGMAP
        path_out = Path(outdir) / (fid + extension)
        Image.fromarray(object.astype(np.uint8)).save(path_out)


def convert_format_from_ndarray(arr: np.ndarray, format: str, path_image: os.PathLike = None):
    if format == NAME_LABELME:
        mask_pred = label(arr)
        for i in range(1, mask_pred.max()+1):
            if np.sum(mask_pred == i) <= 3:  # ignore too small segmentation
                arr[mask_pred == i] = 0
        shapes = []
        for label_index in range(1, mask_pred.max()+1):  # ignore background
            contours = cv2.findContours((arr == label_index).astype(
                np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2]
            for coords_contour in contours:
                if len(coords_contour) > 0:
                    dict_polygon = {"label": str(label_index),
                                    "group_id": None,
                                    "shape_type": "polygon",
                                    "flags": {},
                                    "points": coords_contour[::10, 0, :].tolist()}
                    shapes.append(dict_polygon)
        encoded = base64.b64encode(open(path_image, "rb").read())
        decoded_str = encoded.decode('utf-8')
        return {"version": "4.5.7",
                "flags": {},
                "shapes": shapes,
                "imageData": decoded_str,
                "imagePath": "dummy",
                "imageHeight": arr.shape[0],
                "imageWidth": arr.shape[1]}
    elif format == NAME_SEGMAP:
        return arr
    else:
        raise NotImplementedError("Unknown format {}".format(format))


def convert_format(datahome: str, outdir: str, from_format: str, to_format: str, class_name_to_id: dict = None):

    assert not os.path.exists(outdir), "{} already exists.".format(outdir)

    if from_format.lower() == NAME_LABELME and to_format.lower() == NAME_SEGMAP:

        assert "__ignore__" in class_name_to_id and class_name_to_id["__ignore__"] == -1, "__ignore__ should be set to -1."
        assert "_background_" in class_name_to_id and class_name_to_id["_background_"] == 0, "_background_ should be set to 0."

        for path in Path(datahome).iterdir():
            label_file = labelme.LabelFile(filename=path.__fspath__())
            img = labelme.utils.img_data_to_arr(label_file.imageData)
            lbl, _ = labelme.utils.shapes_to_label(
                img_shape=img.shape,
                shapes=label_file.shapes,
                label_name_to_value=class_name_to_id,
            )
            if not os.path.exists(outdir):
                os.makedirs(outdir, exist_ok=True)
            save_path = os.path.join(outdir, os.path.splitext(path.name)[0])
            labelme.utils.lblsave(save_path, lbl)

    elif from_format.lower() == NAME_SEGMAP and to_format.lower() == NAME_LABELME:
        path_img_dir = Path(datahome) / "image"
        path_mask_dir = Path(datahome) / "mask"
        assert path_img_dir.exists() and path_mask_dir.exists(), "Any of 'image' and 'mask' folders does not exist."

        for path in path_mask_dir.iterdir():
            arr = np.array(Image.open(path))
            cand = glob((path_img_dir / os.path.splitext(path.name)[0]).__fspath__()+"*")
            assert len(cand) == 1
            path_img = cand[0]
            labelme_format = convert_format_from_ndarray(arr, to_format, path_img)
            save_output(labelme_format, to_format, outdir, path.name)
