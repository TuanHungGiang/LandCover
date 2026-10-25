# LoveDA paper-faithful baseline queue

Đây là pipeline dùng cho bảng baseline có thể audit. Nó **không dùng** các decoder rút gọn trong
`baseline_suite/models/zoo.py`.

## Giao thức cố định

- Chỉ train bằng LoveDA `Train`; chỉ đánh giá checkpoint cuối tại iteration 80,000 trên `Val`.
- Không chọn `best checkpoint` bằng Val, vì làm vậy sẽ dùng tập báo cáo để điều chỉnh mô hình.
- Crop 512×512; scale ngẫu nhiên trong `[0.5, 0.75, 1.0, 1.25, 1.5]`; flip ngang, flip dọc và rotation.
- Batch toàn cục hiệu dụng luôn bằng 16. Nếu OOM, micro-batch giảm và gradient accumulation tăng.
- Kiến trúc, optimizer và pretrained backbone lấy từ exact public LoveDA config đã pin commit.
- Chỉ được nạp checkpoint ImageNet của backbone. `load_from`, checkpoint segmentation và checkpoint
  model đã train bởi tác giả đều bị cấm.
- Mỗi run lưu config đã resolve, source-config hash, checkpoint hash, seed, môi trường, raw log và metric.
- Pretrained checkpoint mới tải được băm rồi xóa sau từng model. Checkpoint 80k của chính run không bị xóa.

Chi tiết và link bằng chứng nằm trong `recipes.json`.

## Chế độ strict

- `official_exact`: có public LoveDA config tương ứng trong MMSegmentation: PSPNet-R50 và
  DeepLabv3+-R50.
- `skipped_no_exact_recipe` hoặc `blocked`: thiếu exact public LoveDA recipe, thiếu code, hoặc paper
  không xác định chính xác backbone variant. Queue chỉ ghi lý do rồi bỏ qua; không có cờ để ép chạy.

Kết quả ở đây là **LoveDA Val reproduction**. Table 1 của paper SCSM dùng LoveDA Test qua evaluation
server, nên không được đặt hai con số cạnh nhau như thể cùng split.

## Kaggle

Dùng một trong hai notebook:

- `kaggle_paper_faithful_part1.ipynb`: chỉ train PSPNet-R50 và DeepLabv3+-R50.
- `kaggle_paper_faithful_part2.ipynb`: audit 17 model còn lại và ghi trạng thái `blocked`; không train.

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
5. tên variant/fidelity được đưa nguyên văn vào bảng.

Nên chạy ba seed (ví dụ 2025, 42, 3407) và báo cáo mean ± standard deviation. Paper SCSM không công bố
seed cho Table 1, vì vậy một run đơn chỉ nên được gọi là single-seed reproduction.

