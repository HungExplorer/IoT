#Chung ý tưởng với attempt 2 nhưng đỡ phức tạp hơn 

from counterfit_connection import CounterFitConnection
CounterFitConnection.init('127.0.0.1', 5000)

import io
from counterfit_shims_picamera import PiCamera
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials

# Các thư viện cho việc vẽ
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Khởi tạo camera
camera = PiCamera()
camera.resolution = (640, 480)  # (width, height)
camera.rotation = 0

# Chụp ảnh vào buffer
image_stream = io.BytesIO()
camera.capture(image_stream, 'jpeg')
image_stream.seek(0)

# Lưu file ảnh (tuỳ chọn)
with open('image.jpg', 'wb') as image_file:
    image_file.write(image_stream.read())

# Thông tin Custom Vision
prediction_url = 'https://southeastasia.api.cognitive.microsoft.com/customvision/v3.0/Prediction/ed052184-4478-4262-9ffa-1a35c15c2946/detect/iterations/Iteration2/image'
prediction_key = '2a10664ae7d944ce9cba407aa2fe7b9b'

# Tách endpoint, project_id, iteration_name từ prediction_url
parts = prediction_url.split('/')
endpoint = 'https://' + parts[2]
project_id = parts[6]
iteration_name = parts[9]

# Khởi tạo client
prediction_credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
predictor = CustomVisionPredictionClient(endpoint, prediction_credentials)

# Quay lại đầu buffer để đọc
image_stream.seek(0)
results = predictor.detect_image(project_id, iteration_name, image_stream)

# Ngưỡng lọc
threshold = 0.3

# Kích thước ảnh thực tế
image_width, image_height = camera.resolution

# Lọc những prediction trên ngưỡng và tính tọa độ pixel
predictions = []
for p in results.predictions:
    if p.probability > threshold:
        # chuẩn tỉ lệ
        ln, tn, wn, hn = (
            p.bounding_box.left,
            p.bounding_box.top,
            p.bounding_box.width,
            p.bounding_box.height
        )
        # pixel
        lp = int(ln * image_width)
        tp = int(tn * image_height)
        wp = int(wn * image_width)
        hp = int(hn * image_height)
        predictions.append({
            'tag': p.tag_name,
            'prob': p.probability * 100,
            'left_px': lp, 'top_px': tp,
            'width_px': wp, 'height_px': hp,
            'left_norm': ln, 'top_norm': tn,
            'width_norm': wn, 'height_norm': hn,
        })

# In ra console
for obj in predictions:
    print(f"{obj['tag']}: {obj['prob']:.2f}%")
    print(f"  – Tỉ lệ (L, T, W, H): "
          f"({obj['left_norm']:.3f}, {obj['top_norm']:.3f}, "
          f"{obj['width_norm']:.3f}, {obj['height_norm']:.3f})")
    print(f"  – Pixel  (L, T, W, H): "
          f"({obj['left_px']}, {obj['top_px']}, "
          f"{obj['width_px']}, {obj['height_px']})\n")

# --- PHẦN VẼ HÌNH ---
# Mở ảnh đã lưu
img = Image.open('image.jpg')

fig, ax = plt.subplots()
ax.imshow(img)
ax.axis('off')  # tắt trục

# Vẽ từng bounding box và ghi chú
for obj in predictions:
    rect = patches.Rectangle(
        (obj['left_px'], obj['top_px']),
        obj['width_px'], obj['height_px'],
        fill=False, linewidth=2
    )
    ax.add_patch(rect)
    # Ghi text ngay trên hộp
    text = (f"{obj['tag']} {obj['prob']:.1f}%\n"
            f"L:{obj['left_px']} T:{obj['top_px']}\n"
            f"W:{obj['width_px']} H:{obj['height_px']}")
    ax.text(
        obj['left_px'], obj['top_px'] - 5,
        text, fontsize=8, va='bottom'
    )

plt.show()
