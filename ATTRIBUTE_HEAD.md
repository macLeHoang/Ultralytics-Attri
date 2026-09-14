# Attribute Head — per-box multi-label attributes

Branch: `feat/box-attributes`

Standard YOLO gives every box exactly **one** class. The attribute head keeps that single class and adds **K independent yes/no attributes per box**, so one detection can be:

```
person  0.91   black_hair  short  wearing_hat
car     0.87   red  sedan
```

It works for **detect** and **instance segment**, for both **YOLO26** and **YOLO11/v8** heads, across train, val, predict and export.

---

## 1. Quick start

```python
from ultralytics import YOLO

# Build an attribute model and load COCO-pretrained weights for everything except the new attribute branch
model = YOLO("yolo26n-seg-attr.yaml").load("yolo26n-seg.pt")
model.train(data="my_attrs.yaml", epochs=100, imgsz=640)

r = model("image.jpg")[0]
r.attr_names  # ['black_hair', 'short', 'wearing_hat', 'red', 'sedan', 'truck']
r.attributes.data  # (N, K) probabilities, one row per box
r.summary()  # per box: name, class, confidence, box, (segments), attributes
```

CLI:

```bash
yolo segment train model=yolo26n-seg-attr.yaml pretrained=yolo26n-seg.pt data=my_attrs.yaml epochs=100
yolo segment predict model=runs/segment/train/weights/best.pt source=images/
```

---

## 2. Dataset format

### 2.1 `data.yaml`

```yaml
path: /data/my_dataset
train: images/train
val: images/val

names:
  0: person
  1: car

# One global attribute list. Its order is the column order in the label files.
attributes: [black_hair, short, wearing_hat, red, sedan, truck] # list, or {0: black_hair, 1: short, ...}

# Optional: which classes each attribute applies to (class names or ids).
# Attributes not listed here apply to every class.
attr_classes:
  black_hair: [person]
  short: [person]
  wearing_hat: [person]
  red: [person, car] # shared: shirt color for person, body color for car
  sedan: [car]
  truck: [car]
```

`attr_classes` builds an internal `(nc, K)` applicability mask (`attr_mask`) that is used in three places:

- **Loss:** attributes that don't apply to a box's class are never trained for that box.
- **Metrics:** they are excluded from attribute AP.
- **Predict:** they are reported as `0`.

### 2.2 Label files

Append **exactly K values** to the end of **every** label line. Each value is one of:

| Value | Meaning                                           |
| ----- | ------------------------------------------------- |
| `1`   | attribute present                                 |
| `0`   | attribute absent                                  |
| `-1`  | unknown / not labeled → ignored in loss & metrics |

**Detect** (`cls cx cy w h` + K attributes):

```
0 0.31 0.52 0.10 0.40   1 1 0  0 -1 -1
1 0.70 0.60 0.30 0.20  -1 -1 -1 1  1  0
```

**Segment** (`cls x1 y1 ... xn yn` + K attributes; the polygon can have any length):

```
0 0.30 0.32 0.35 0.31 0.36 0.70 0.29 0.72   1 1 0  0 -1 -1
1 0.55 0.50 0.85 0.50 0.86 0.71 0.54 0.70  -1 -1 -1 1  1  0
```

Rules checked at load time. A file that breaks them is reported as corrupt and skipped:

- every line needs at least `5 + K` values;
- attribute values must be `-1`, `0` or `1`;
- the usual YOLO checks (normalized coordinates, class < nc, and so on) apply to the part before the attributes.

> **Cache gotcha:** Ultralytics validates `labels/*.cache` by file paths and **total size**, not contents. Flipping an attribute `0 → 1` keeps the file size the same, so the old cache is reused. **Delete `labels/*.cache` after editing attribute values.** Changing the number of attributes K invalidates the cache automatically.

---

## 3. Models

Four new model configs. Each is its base config with the head swapped and an `na` field added. `na` is overridden automatically from the length of `attributes` in `data.yaml`.

| Config                  | Head            | Base / pretrained weights |
| ----------------------- | --------------- | ------------------------- |
| `yolo26n-attr.yaml`     | `DetectAttr`    | `yolo26n.pt`              |
| `yolo26n-seg-attr.yaml` | `Segment26Attr` | `yolo26n-seg.pt`          |
| `yolo11n-attr.yaml`     | `DetectAttr`    | `yolo11n.pt`              |
| `yolo11n-seg-attr.yaml` | `SegmentAttr`   | `yolo11n-seg.pt`          |

The `n` scale letter can be swapped for any scale: `yolo26s-seg-attr.yaml`, `yolo11m-attr.yaml`, and so on.

When pretrained weights are loaded, everything transfers except the new attribute branch, which starts randomly initialized. The log shows e.g. `Transferred 708/864 items from pretrained weights`.

Using a data.yaml with `attributes` on a normal model, or an attribute model on data without `attributes`, stops training with a clear `ValueError` pointing to these configs.

---

## 4. Training

Nothing new is required beyond the data and model above. One new hyperparameter:

| Arg    | Default | Meaning                                                                |
| ------ | ------- | ---------------------------------------------------------------------- |
| `attr` | `1.0`   | Attribute loss gain. It sits next to `box`, `cls`, `dfl`, `pose`, etc. |

The training log shows an extra column, `attr_loss`, and `results.csv` / `results.png` get `train/attr_loss`, `val/attr_loss`, `metrics/attr_mAP(A)` and `metrics/attr_F1(A)`.

**Tips**

- `attr_loss` starts around `0.69` (`ln 2`, a 50/50 guess). If it stays near that value, the attribute branch isn't learning, and the usual cause is a **learning rate that is too small**:
    - For short trainings (< 10,000 iterations), `optimizer=auto` picks AdamW with `lr = 0.01 / (4 + nc)`. That is ~0.0017 for 2 classes but only ~0.00012 for 80 classes.
    - Measured on this branch with 80 classes: with the auto learning rate, attribute probabilities stayed near 0.5 after 60 epochs on a tiny dataset. With `optimizer=SGD lr0=0.01`, the same head fit the attributes almost perfectly in 150 steps (§11).
    - If `attr_loss` stays flat, set the optimizer explicitly (`optimizer=SGD lr0=0.01` or `optimizer=AdamW lr0=0.001`), and/or raise `attr` (e.g. `attr=2.0`).
- Fine-grained attributes (hair color, clothing) need pixels. Train at a larger `imgsz` if targets are small.
- Mutually exclusive attributes (`sedan`/`truck`/`bus`) are trained as independent sigmoids. Pick the max within the group at inference if you need exactly one.

---

## 5. Validation & metrics

```python
metrics = YOLO("best.pt").val(data="my_attrs.yaml")
metrics.results_dict["metrics/attr_mAP(A)"]
metrics.attr_results  # (p, r, f1, ap, attribute_index) per attribute
```

The console prints an attribute table under the usual box/mask table. Per-attribute rows appear when `verbose=True`:

```
             Attribute          P          R         F1         AP
                   all      0.427          1      0.576      0.838
             is_person      0.455          1      0.625      0.622
                  wide      0.625          1      0.769      0.898
```

How the metrics are computed:

1. **Matching:** each ground-truth box is paired with its **highest-confidence same-class prediction with IoU ≥ 0.5**, one-to-one. Missed ground truths and false-positive boxes are excluded, so attribute metrics answer "given a correct detection, how good are its attributes?".
2. **Filtering:** for each matched pair and attribute, entries with target `-1` or not applicable to the class (`attr_classes`) are dropped.
3. **Scoring:** per-attribute ranking **AP** over the remaining probabilities. **P/R/F1** are taken at the confidence threshold with the best mean F1, using the same `ap_per_class` as box metrics.
4. **`best.pt` selection:** `fitness = usual box (+ mask) fitness + 0.1 × attr_mAP`.

---

## 6. Prediction — `Results` API

| Field                                   | Type                           | Notes                                                                                |
| --------------------------------------- | ------------------------------ | ------------------------------------------------------------------------------------ |
| `r.attributes`                          | `BaseTensor` (N, K) or `None`  | Sigmoid probabilities. Non-applicable attributes for the box's class are `0`.         |
| `r.attributes.data`                     | `torch.Tensor` / `np.ndarray`  | Raw values. `.cpu()`, `.numpy()`, `.shape` work.                                     |
| `r.attr_names`                          | `list[str]`                    | Column names, in `data.yaml` order.                                                  |
| `r.summary()` / `to_json()` / `to_df()` | per box `"attributes": {...}`  | `{name: probability}` for every attribute.                                           |
| `r.plot()`                              | image                          | Box label gets the names of attributes with probability > 0.5, e.g. `person 0.91 black_hair short`. |
| `r[idx]`                                | `Results`                      | Indexing/slicing keeps attributes aligned with boxes and masks.                      |

Example — keep only people with black hair:

```python
r = model("street.jpg")[0]
k = r.attr_names.index("black_hair")
keep = (r.boxes.cls == 0) & (r.attributes.data[:, k] > 0.5)
people_black_hair = r[keep.nonzero().flatten()]
```

---

## 7. Export & deployment

```python
YOLO("best.pt").export(format="onnx")  # also engine, openvino, ...
YOLO("best.onnx", task="segment")("image.jpg")[0].attributes  # works like .pt
```

- The attribute names and applicability mask are written to the export metadata (`attributes`, `attr_mask`), so Ultralytics can predict from exported files with no extra config.
- Attributes are part of the model graph, already sigmoided, and always come **last**:

| Output                            | Layout per anchor / detection                                                               |
| --------------------------------- | ------------------------------------------------------------------------------------------- |
| Raw detect `(B, 4+nc+K, A)`       | `cx, cy, w, h, class scores (nc), attributes (K)`                                           |
| Raw segment `(B, 4+nc+32+K, A)`   | `cx, cy, w, h, class scores (nc), mask coefs (32), attributes (K)`, plus protos output       |
| After NMS / NMS-free `(B, N, ·)`  | `x1, y1, x2, y2, conf, cls, [mask coefs (32)], attributes (K)`                              |

The export log prints the actual output shape. A custom parser such as DeepStream only needs to read the last K channels; no parser is included on this branch.

Verified: ONNX export and ONNX Runtime prediction give the same attributes as PyTorch to within ~4e-4. Other formats use the same graph but were not tested.

---

## 8. How it works

```
backbone/neck features (P3, P4, P5)
        │
        ├── cv2 ─► box distribution           (unchanged)
        ├── cv3 ─► class logits (nc)          (unchanged, single-label)
        ├── cv4 ─► mask coefficients (32)     (segment only, unchanged)
        └── cv5 ─► attribute logits (K)       (NEW, same structure as cv3; one2one_cv5 for YOLO26 end2end)
```

- **Head (`AttrHead` mixin):** adds `cv5`, and `one2one_cv5` for end2end models. It appends `attrs` to the training outputs and `sigmoid(attrs)` as the last K channels of the inference output. Top-k/NMS, end2end postprocess and the exporter's embedded NMS already carry trailing channels, so none of them needed changes.
- **Loss (`AttrLoss` mixin):** runs the normal task-aligned assigner once and reuses its `fg_mask` / `target_gt_idx`. Each foreground anchor gets its assigned ground truth's attribute vector, and the loss is
  `BCEWithLogits(pred, target) × valid`, averaged over valid entries and multiplied by `attr`, where `valid = foreground & target != -1 & attr_mask[gt_class]`.
  With YOLO26 (`E2ELoss`) the same loss is applied to both the one-to-many and one-to-one branches.
- **Data:** attributes are stored per instance and move through **every** augmentation that filters, reorders or concatenates instances. That includes the area sort used for overlapping masks, where a missed reorder would silently attach attributes to the wrong masks. `Format` asserts that the attribute count equals the instance count.

Why not simply write the same box once per label? The assigner resolves ties between identical boxes with `argmax`, so each anchor learns only one of the labels and treats the rest as negatives. NMS and mAP would also treat `black_hair` as a separate object.

---

## 9. What changed

| File                                                                  | Change                                                                                                                                                                                                      |
| --------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ultralytics/data/utils.py`                                           | `verify_image_label` removes the trailing K attribute columns before box/segment parsing and validates them. `check_det_dataset` validates `attributes` / `attr_classes` and builds `attr_mask`.            |
| `ultralytics/data/dataset.py`                                         | Labels store `attributes`. The cache hash includes K. `collate_fn` concatenates attributes.                                                                                                                  |
| `ultralytics/data/base.py`                                            | The `classes=` filter also filters attributes.                                                                                                                                                              |
| `ultralytics/data/augment.py`                                         | Attributes follow instances in `Mosaic`, `MixUp`, `CutMix`, `RandomPerspective`, `CopyPaste`, `Albumentations` and `Format` (including the overlap-mask area sort).                                          |
| `ultralytics/nn/modules/head.py`, `nn/modules/__init__.py`            | New `AttrHead` mixin and the `DetectAttr`, `SegmentAttr`, `Segment26Attr` heads.                                                                                                                             |
| `ultralytics/nn/tasks.py`                                             | `parse_model` resolves `na` and registers the new heads. `DetectionModel` / `SegmentationModel` accept `na`. `init_criterion` picks the attribute losses.                                                    |
| `ultralytics/cfg/models/26/yolo26-attr.yaml`, `yolo26-seg-attr.yaml`  | New model configs.                                                                                                                                                                                          |
| `ultralytics/cfg/models/11/yolo11-attr.yaml`, `yolo11-seg-attr.yaml`  | New model configs.                                                                                                                                                                                          |
| `ultralytics/utils/loss.py`                                           | New `AttrLoss` mixin, `v8DetectAttrLoss`, `v8SegmentAttrLoss`.                                                                                                                                              |
| `ultralytics/cfg/default.yaml`, `cfg/__init__.py`                     | New `attr` loss gain (float arg).                                                                                                                                                                           |
| `ultralytics/models/yolo/detect/train.py`, `segment/train.py`         | Pass K to the model, attach `attributes` / `attr_mask` to the model, and check that data and head match.                                                                                                     |
| `ultralytics/models/yolo/detect/val.py`                               | NMS uses the real class count when attributes exist, splits attribute channels, matches predictions to ground truth, prints the attribute table.                                                            |
| `ultralytics/utils/metrics.py`                                        | `DetMetrics` computes attribute AP/P/R/F1, adds `attr_mAP` / `attr_F1` to `results_dict`, and adds `0.1 × attr_mAP` to fitness. Inherited by `SegmentMetrics`; tasks without attributes are unchanged.       |
| `ultralytics/models/yolo/detect/predict.py`, `segment/predict.py`     | NMS class count, `get_attributes()` (applies `attr_mask`), mask coefficients exclude the attribute channels.                                                                                                |
| `ultralytics/engine/results.py`                                       | `Results.attributes`, `Results.attr_names`, attribute names in `plot()` labels, `attributes` in `summary()`.                                                                                                |
| `ultralytics/engine/exporter.py`, `nn/backends/base.py`, `nn/backends/pytorch.py` | Attribute names and mask written to / read from export metadata.                                                                                                                                |
| `tests/test_attributes.py`                                            | New tests: augmentation alignment (detect + segment) and train → val → predict (YOLO26 detect, YOLO11 segment).                                                                                             |

Datasets without `attributes` go through the same code paths as before.

---

## 10. Limitations & notes

- **Tasks:** detect and instance segment only. Pose, OBB, semantic, RT-DETR and YOLOE don't support attributes.
- **Attribute metrics** only cover correctly detected boxes (§5). Detection quality is measured separately by box/mask mAP.
- **Tracking** (`model.track`) keeps attributes aligned with tracked boxes by design, but this isn't covered by tests, and there is no temporal smoothing of attributes across frames.
- **Exports** other than ONNX were not tested.
- **Deployment parsers** (DeepStream, etc.) are not included; see the output layout in §7.
- **Label conversion:** annotation tools that export "one box per label" need a small conversion script that merges those rows into one line with an attribute vector.

---

## 11. Tests

Run the tests from a scratch directory, because the test session cleanup deletes some files in the current directory:

```bash
cd /tmp/scratch && PYTHONPATH=/path/to/ultralytics python -m pytest /path/to/ultralytics/tests/test_attributes.py -v
```

Verified on this branch:

- `tests/test_attributes.py`: 4 tests passed.
- Augmentation alignment: ~6,000 augmented instances with 0 misaligned, and a deliberately injected misalignment bug was caught.
- Loss targets match the assigner's ground truth for every foreground anchor. `-1` values and non-applicable attributes add nothing to the loss.
- Full train → val → predict → ONNX export → ONNX predict for YOLO26 detect.
- **Learning check** (YOLO26n from `yolo26n.pt`, 4 images, fixed batch, SGD lr 0.01, 150 steps): `attr_loss` fell from 0.82 to 0.0018. Through the normal predictor, `is_person` averaged **1.000 on people vs 0.003 on other classes**. `big`, which applies only to people, was 0 on every other class. This shows the head, loss and prediction path work; it doesn't measure how well attributes generalize to new images.
- Short runs with `optimizer=auto` and 80 classes (60 epochs, 4 images): attribute mAP rose (detect 0.76 → 0.94, segment 0.40 → 0.60), but probabilities stayed near 0.5 because of the tiny auto learning rate (see §4 Tips).
- Not yet verified: attribute quality on a real dataset.
