## Keypoints Studio (YOLO Pose Annotator)

Desktop GUI + CLI to auto-annotate and edit **keypoint (pose) datasets** using Ultralytics YOLO pose models.

- **Auto-annotate** folders with a YOLO pose model
- **Preview / Edit**: zoom & pan, drag/add/delete keypoints, resize/delete bboxes
- **Custom keypoint schema**: reorder YOLO keypoints and add custom keypoints (e.g., Trachea / Forehead / FootIndex)
- **Save labels next to images**: `image.jpg` → `image.txt` (YOLO pose format)
- **Labeled-folder mode**: if `keypoints.txt` exists in a folder, auto-annotate is disabled and Predict/Preview loads existing labels (no model inference)

### Screenshots

Put your screenshots in the repository under `images/` and they will show here on GitHub.

- Main window:

![Main window](./images/GUI.png)

- Mapping dialog:

![Keypoint mapping](./images/custom_keypoints.png)

> If images are not showing, confirm the files exist in the repo under `images/` and the filenames match exactly.

## Requirements

- Windows 10/11
- Anaconda/Miniconda
- NVIDIA GPU (optional; CPU works)

## Install (Conda)

Open **Anaconda Prompt**:

```bat
conda create -n torch_gpu python=3.10 -y
conda activate torch_gpu
```

### (Optional) GPU PyTorch (CUDA)

```bat
conda activate torch_gpu
conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

Verify:

```bat
python -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('gpus', torch.cuda.device_count())"
```

## Install this project (editable)

From the project folder:

```bat
conda activate torch_gpu
cd /d "C:\Users\GS\Downloads\cus_keypoints"
python -m pip install -U pip
python -m pip install -e .
```

## Run the GUI

```bat
conda activate torch_gpu
pose-annotator-gui
```

### GUI workflow

1. Click **Select Images Folder…**
2. If the folder is unlabeled: click **Predict / Preview** (runs model) or **Auto-annotate folder**
3. Edit as needed:
   - Mouse wheel = zoom, drag = pan
   - Drag keypoints
   - **Add keypoint**: set `KP#` + `Person`, enable Add mode, then click on image
   - Select keypoint/bbox and press **Delete**
4. Click **Save label** (or enable **Auto save** and use A/D navigation)

### Output files (next to each image)

- `image.txt`: YOLO pose label file
- `keypoints.txt`: written on Predict/Preview (image size + keypoint list) and also used to mark a folder as “labeled”

## Run the CLI (batch auto-annotation)

```bat
conda activate torch_gpu
pose-annotator "C:\path\to\images" --device 0 -v
```

Notes:
- Use `--device cpu` for CPU inference.
- CLI output defaults to a separate labels folder (see `--labels-dir`). The GUI writes labels next to images.

## Label format (YOLO pose)

One line per person:

`class x_center y_center width height x1 y1 v1 ... xK yK vK`

- All coordinates are normalized to `[0,1]`
- Visibility `v`:
  - `0` = missing / removed
  - `2` = present
- With a custom schema, `K` can be **> 17**

Reference: `https://docs.ultralytics.com/datasets/pose/`

## Troubleshooting

### CUDA not available / `torch.cuda.is_available(): False`

```bat
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Reinstall CUDA PyTorch (recommended):

```bat
conda activate torch_gpu
conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

### `WinError 32` during reinstall (exe locked)

Close the GUI and reinstall:

```bat
python -m pip install -e .
```

