# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import shutil

import pytest

from tests import SOURCE
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data import build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import ASSETS, YAML

# (class, x1, y1, x2, y2, attributes [is_person, wide, big]); -1 marks an unlabeled attribute
OBJECTS = [
    (0, 0.05, 0.2, 0.3, 0.9, [1, 0, 1]),
    (1, 0.35, 0.3, 0.95, 0.7, [0, 1, -1]),
    (0, 0.7, 0.1, 0.8, 0.4, [1, 0, 0]),
]


def make_attr_dataset(root, task):
    """Create a two-image detect or segment dataset whose label rows end with three attribute columns."""
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        for im in (SOURCE, ASSETS / "zidane.jpg"):
            shutil.copy(im, root / "images" / split / im.name)
            rows = []
            for c, x1, y1, x2, y2, attrs in OBJECTS:
                geom = (
                    [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]
                    if task == "detect"
                    else [x1, y1, x2, y1, x2, y2, x1, y2]
                )
                rows.append(" ".join(map(str, [c, *geom, *attrs])))
            (root / "labels" / split / f"{im.stem}.txt").write_text("\n".join(rows))
    data = {"path": str(root), "train": "images/train", "val": "images/val", "names": ["person", "car"]}
    YAML.save(
        root / "data.yaml", {**data, "attributes": ["is_person", "wide", "big"], "attr_classes": {"big": ["person"]}}
    )
    return root / "data.yaml"


@pytest.mark.parametrize("task", ["detect", "segment"])
def test_attributes_follow_augmentation(tmp_path, task):
    """Test that per-box attributes stay aligned with their instances through mosaic, mixup and copy-paste."""
    data = check_det_dataset(make_attr_dataset(tmp_path, task))
    assert data["attr_mask"] == [[1, 1, 1], [1, 1, 0]]
    hyp = {"task": task, "imgsz": 160, "mosaic": 1.0, "mixup": 1.0, "cutmix": 1.0, "copy_paste": 1.0, "degrees": 20}
    dataset = build_yolo_dataset(get_cfg(overrides=hyp), data["train"], 2, data, mode="train")
    for i in range(20):
        sample = dataset[i % len(dataset)]
        cls = sample["cls"].view(-1)
        assert sample["attributes"].shape == (len(cls), 3)
        assert (sample["attributes"][:, 0] == (cls == 0).float()).all()  # is_person must follow its instance


@pytest.mark.parametrize("cfg", ["yolo26n-attr.yaml", "yolo11n-seg-attr.yaml"])
def test_attributes_train_val_predict(tmp_path, cfg):
    """Test training, validating and predicting with per-box attribute heads."""
    data = make_attr_dataset(tmp_path / "data", "segment" if "seg" in cfg else "detect")
    model = YOLO(cfg)
    model.train(data=data, epochs=1, imgsz=64, batch=2, workers=0, project=tmp_path, name="train", plots=False)
    metrics = YOLO(model.trainer.best).val(data=data, imgsz=64, batch=2, workers=0, plots=False)
    assert "metrics/attr_mAP(A)" in metrics.results_dict
    result = YOLO(model.trainer.best)(SOURCE, imgsz=64, conf=0.0)[0]
    assert result.attr_names == ["is_person", "wide", "big"]
    assert result.attributes.shape == (len(result), 3)
