# Usage

```bash
cargo run -- "<model name>" "<path to image of cropped face>"
```

## Pre-requisites

You will need to download the model from https://github.com/av-savchenko/face-emotion-recognition

```bash
wget https://github.com/av-savchenko/face-emotion-recognition/blob/main/models/affectnet_emotions/enet_b2_8_best.pt
```

Then you will need to convert the model to a format that can be used by the `tch-rs` library.

```bash
python3 utils/convert_to_jit.py enet_b2_8_best.pt enet_b2_8_best_jit.pt # This is now your model name
```

And crop some faces from images using the following command:

```bash
python3 utils/crop_faces.py "<path to image>" "<path to outpu>t" ["<path to more outputs if multiple faces exist in image>"]
```
