# LoveDA unified baseline suite

Thư mục này cung cấp **một pipeline train/evaluate chung** cho 19 tên model trong bảng baseline.

## Quy tắc dữ liệu bắt buộc

- Huấn luyện: chỉ `LoveDA/Train`.
- Đánh giá và chọn checkpoint: chỉ `LoveDA/Val`.
- `LoveDA/Test` không được loader chấp nhận và `Test.zip` không được giải nén.
- Nhãn LoveDA gốc `1..7` được ánh xạ thành `0..6`; nhãn `0` (no-data) thành `ignore_index=255`.
- mIoU được tính đều trên bảy lớp: background, building, road, water, barren, forest, agriculture.

## Chuẩn bị

Môi trường hiện tại đã có PyTorch và timm. Nếu tạo môi trường mới:

```powershell
cd D:\landcover\baseline_model
python -m pip install -r requirements.txt
python prepare_data.py
```

`prepare_data.py` chỉ đọc:

- `D:\landcover\data\loveda\Train.zip`
- `D:\landcover\data\loveda\Val.zip`

và giải nén vào `D:\landcover\data\loveda\extracted`.

## Train chung

```powershell
python train.py --model pspnet --epochs 80 --batch-size 4 --amp
python train.py --model deeplabv3plus --epochs 80 --batch-size 4 --amp
python train.py --model swin_upernet --epochs 80 --batch-size 2 --accumulate 2 --amp
```

Chạy tuần tự toàn bộ model:

```powershell
.\train_all.ps1 -Epochs 80 -BatchSize 4
```

Danh sách tên hợp lệ:

```text
pspnet, deeplabv3plus, danet, semantic_fpn, farseg, ocrnet, lanet,
isnet, segmenter, swin_upernet, manet, flanet, convnext, poolformer,
biformer, efficientvit, ddp, logcanpp, scsm
```

Để xem backbone và mức độ fidelity:

```powershell
python list_models.py
```

Kết quả nằm trong `runs/<model>/`:

- `best.pt`: checkpoint có mIoU Val tốt nhất;
- `last.pt`: checkpoint epoch cuối;
- `history.json`: train loss và metric Val từng epoch;
- `val_metrics.json`: kết quả cuối trên LoveDA Val;
- `config.json`: toàn bộ tham số lần chạy.

## Test checkpoint trên LoveDA Val

```powershell
python train.py `
  --mode test `
  --model pspnet `
  --checkpoint runs\pspnet\best.pt
```

Không có chế độ test trên `Test`, vì nhãn Test không được công bố và phải nộp lên evaluation server.

## Mức độ tương thích với paper

Pipeline chung giữ cùng data split, loss, optimizer và metric để so sánh công bằng. Các model dùng backbone `timm` và decoder chuẩn hóa trong `baseline_suite/models/zoo.py`.

- Nhóm có cấu trúc chuẩn hóa trực tiếp: PSPNet, DeepLabv3+, DANet, Semantic FPN, FarSeg, OCRNet, LANet, ISNet, Segmenter, Swin-UPerNet, MANet, FLANet, ConvNeXt-UPerNet, PoolFormer-UPerNet và EfficientViT-UPerNet.
- `BiFormer`, `DDP` và `LOGCAN++` là bản paper-derived gọn để đi qua cùng trainer; dùng `sources.json` để đối chiếu mã tác giả khi cần tái lập đúng tuyệt đối.
- Tính đến thời điểm tạo suite, repository tác giả **chưa phát hành source/checkpoint SCSM**. `scsm` ở đây là reproduction từ mô tả bài báo và được đánh dấu `paper-derived-unofficial`; không được trình bày như kết quả official.

Các URL nguồn chính thức nằm trong `sources.json`. Bản `rssegmentation` có sẵn trên máy được lưu trong `official_sources/rssegmentation_offline` để đối chiếu LOGCAN++ và thông tin SCSM.

## Thiết lập so sánh khuyến nghị

- `crop_size=512`, `seed=42`, AdamW, cosine schedule.
- Cùng số epoch và effective batch size cho mọi model.
- Không bật `--pretrained` nếu máy không có Internet/cache checkpoint.
- Với GPU ít VRAM, giảm `--batch-size` và tăng `--accumulate` tương ứng.


## Kaggle: tự tối ưu batch size và tiếp tục khi OOM

Mở `kaggle_train_all.ipynb` trên Kaggle, bật GPU Accelerator và Run All. Notebook dùng trực tiếp:

- `/kaggle/input/datasets/alienxc137/loveda-satellite-images-semantic-segmentation/Train`
- `/kaggle/input/datasets/alienxc137/loveda-satellite-images-semantic-segmentation/Val`

Với từng model, notebook chạy một optimizer step thử nghiệm, tìm batch size nguyên lớn nhất bằng exponential/binary search, bắt CUDA OOM, dọn VRAM và tự giảm batch. OOM trong lúc train cũng được bắt để session và các model kế tiếp tiếp tục chạy. Kết quả được lưu dần trong `/kaggle/working/loveda_baseline_runs`.
