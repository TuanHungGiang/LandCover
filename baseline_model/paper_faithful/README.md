# LoveDA official-architecture, backbone-pretrained baseline queue

Đây là pipeline dùng cho bảng baseline có thể audit. Nó **không dùng** các decoder rút gọn trong
`baseline_suite/models/zoo.py`.

## Giao thức cố định

- Chỉ train bằng LoveDA `Train`; chỉ đánh giá checkpoint cuối tại iteration 80,000 trên `Val`.
- Không chọn `best checkpoint` bằng Val, vì làm vậy sẽ dùng tập báo cáo để điều chỉnh mô hình.
- Crop 512×512; scale ngẫu nhiên trong `[0.5, 0.75, 1.0, 1.25, 1.5]`; flip ngang, flip dọc và rotation.
- Batch toàn cục hiệu dụng luôn bằng 16. Nếu OOM, micro-batch giảm và gradient accumulation tăng.
- Kiến trúc, optimizer và ImageNet-pretrained backbone lấy từ public config đã pin commit.
- Chỉ backbone được nạp pretrained. Segmentation head/decoder khởi tạo ngẫu nhiên; `load_from` và
  mọi checkpoint segmentation/task đều bị cấm.
- Mỗi run lưu config đã resolve, source-config hash, checkpoint hash, seed, môi trường, raw log và metric.
- Pretrained backbone vừa tải được ghi checksum rồi xóa sau mỗi model. Checkpoint 80k do chính run
  tạo ra được giữ lại.

Chi tiết và link bằng chứng nằm trong `recipes.json`.

## Chế độ backbone-pretrained

- `runnable_backbone_pretrained`: có kiến trúc executable và backbone initialization trong model zoo
  đã pin. Hiện gồm PSPNet-R50,
  DeepLabv3+-R50, DANet-R50, Semantic FPN-R50, OCRNet-HRNetW48, Segmenter-ViT-B,
  Swin-T-UPerNet, ConvNeXt-Tiny-UPerNet và PoolFormer-S12.
- `blocked`: chưa tích hợp được code chính xác hoặc code/config không được công bố. Queue bỏ qua,
  không tự viết implementation gần đúng.

Kết quả ở đây là **LoveDA Val reproduction**. Table 1 của paper SCSM dùng LoveDA Test qua evaluation
server, nên không được đặt hai con số cạnh nhau như thể cùng split.

## Kaggle

Dùng một trong hai notebook:

- `kaggle_paper_faithful_part1.ipynb`: PSPNet, DeepLabv3+, DANet, Semantic FPN, OCRNet.
- `kaggle_paper_faithful_part2.ipynb`: Segmenter, SwinUperNet, ConvNeXt-UPerNet, PoolFormer.

Notebook tự dùng:

```text
/kaggle/input/datasets/alienxc137/loveda-satellite-images-semantic-segmentation/Train
/kaggle/input/datasets/alienxc137/loveda-satellite-images-semantic-segmentation/Val
```

Bạn có thể sửa `MODELS` còn một model để mỗi Kaggle session chỉ chạy một model. Nếu session bị ngắt,
giữ thư mục output của phiên trước làm Kaggle Dataset rồi đặt nó lại đúng đường dẫn output; queue sẽ
resume từ `last_checkpoint` và bỏ qua model đã `complete`.

## Output tối thiểu cho một hàng baseline

Chỉ chấp nhận run có:

1. `state == "complete"` trong `status.json`;
2. checkpoint đúng `iter_80000.pth` cùng SHA-256;
3. `generated_config.py` và `provenance.json`;
4. `test_val.log` và metrics đọc từ lần test checkpoint cuối;
5. `initialization=backbone_pretrained` và `backbone_pretrained_fields` trong provenance;
6. tên variant/fidelity được đưa nguyên văn vào bảng.

Nên chạy ba seed (ví dụ 2025, 42, 3407) và báo cáo mean ± standard deviation. Paper SCSM không công bố
seed cho Table 1, vì vậy một run đơn chỉ nên được gọi là single-seed reproduction. Khác biệt GPU và
micro-batch phải được ghi trong phần experimental settings.

