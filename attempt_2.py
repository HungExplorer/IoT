#attempt 2: bouding box with pixel information

from counterfit_connection import CounterFitConnection
CounterFitConnection.init('127.0.0.1', 5000)

import io
from counterfit_shims_picamera import PiCamera
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- 1. Chụp ảnh từ camera ---
camera = PiCamera()
camera.resolution = (640, 480)
camera.rotation = 0

image_stream = io.BytesIO()
camera.capture(image_stream, 'jpeg')
image_stream.seek(0)

# Lưu tạm ảnh
with open('image.jpg', 'wb') as f:
    f.write(image_stream.read())

# --- 2. Gửi lên Custom Vision để nhận kết quả ---
prediction_url = 'https://southeastasia.api.cognitive.microsoft.com/customvision/v3.0/Prediction/ed052184-4478-4262-9ffa-1a35c15c2946/detect/iterations/Iteration2/image'
prediction_key = '2a10664ae7d944ce9cba407aa2fe7b9b'

parts = prediction_url.split('/')
endpoint = 'https://' + parts[2]
project_id = parts[6]
iteration_name = parts[9]

credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
predictor = CustomVisionPredictionClient(endpoint, credentials)

image_stream.seek(0)
results = predictor.detect_image(project_id, iteration_name, image_stream)

# --- 3. Phân tích kết quả ---
threshold = 0.3
image_width, image_height = camera.resolution

predictions = []
for p in results.predictions:
    if p.probability > threshold:
        ln, tn, wn, hn = p.bounding_box.left, p.bounding_box.top, p.bounding_box.width, p.bounding_box.height
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

# --- 4. In kết quả ra console ---
for obj in predictions:
    print(f"{obj['tag']}: {obj['prob']:.2f}%")
    print(f"  – Tỉ lệ (L, T, W, H): ({obj['left_norm']:.3f}, {obj['top_norm']:.3f}, {obj['width_norm']:.3f}, {obj['height_norm']:.3f})")
    print(f"  – Pixel  (L, T, W, H): ({obj['left_px']}, {obj['top_px']}, {obj['width_px']}, {obj['height_px']})\n")

# --- 5. Vẽ hình ảnh kèm chú thích chi tiết ---
img = Image.open('image.jpg')
fig, ax = plt.subplots(figsize=(10, 8))
ax.imshow(img)

# Vẽ từng bounding box + mũi tên + label
for obj in predictions:
    lp, tp, wp, hp = obj['left_px'], obj['top_px'], obj['width_px'], obj['height_px']

    # Bounding box
    rect = patches.Rectangle((lp, tp), wp, hp,
                             linewidth=2, edgecolor='lime', linestyle='dashed',
                             facecolor='none')
    ax.add_patch(rect)

    # Ghi nhãn trên hộp
    tag = obj['tag']
    prob = obj['prob']
    ax.text(lp, tp - 5,
            f"{tag} {prob:.1f}%",
            color='white', fontsize=9, weight='bold',
            bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.2'))

    # Mũi tên chiều cao
    ax.annotate('', xy=(lp, tp), xytext=(lp, tp + hp),
                arrowprops=dict(arrowstyle='<->', color='yellow'))
    ax.text(lp + 5, tp + hp // 2, f"H: {hp}", va='center', fontsize=8, color='yellow')

    # Mũi tên chiều rộng
    ax.annotate('', xy=(lp, tp + hp), xytext=(lp + wp, tp + hp),
                arrowprops=dict(arrowstyle='<->', color='cyan'))
    ax.text(lp + wp // 2, tp + hp + 10, f"W: {wp}", ha='center', fontsize=8, color='cyan')

    # Mũi tên từ góc ảnh tới (left, top)
    ax.annotate('', xy=(lp, 0), xytext=(lp, tp),
                arrowprops=dict(arrowstyle='<->', color='magenta'))
    ax.text(lp + 5, tp // 2, f"T: {tp}", fontsize=8, color='magenta', va='center')

    ax.annotate('', xy=(0, tp), xytext=(lp, tp),
                arrowprops=dict(arrowstyle='<->', color='orange'))
    ax.text(lp // 2, tp + 5, f"L: {lp}", fontsize=8, color='orange', ha='center')

    # Hiển thị thông số chuẩn hóa góc phải
    info = (
        f"{tag} {prob:.1f}%\n"
        f"L: {obj['left_norm']:.2f} | {lp}\n"
        f"T: {obj['top_norm']:.2f} | {tp}\n"
        f"W: {obj['width_norm']:.2f} | {wp}\n"
        f"H: {obj['height_norm']:.2f} | {hp}"
    )
    ax.text(image_width + 20, 30 + 90 * predictions.index(obj),
            info, fontsize=9, family='monospace',
            bbox=dict(facecolor='white', alpha=0.8))

# Kích thước canvas đủ rộng để chứa text bên phải
ax.set_xlim(0, image_width + 200)
ax.set_ylim(image_height, 0)
ax.axis('off')
plt.tight_layout()
plt.show()
