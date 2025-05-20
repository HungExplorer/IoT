from counterfit_connection import CounterFitConnection
CounterFitConnection.init('127.0.0.1', 5000)

import io
from counterfit_shims_picamera import PiCamera

from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials

from PIL import Image, ImageDraw, ImageColor

# Chụp ảnh từ camera
camera = PiCamera()
camera.resolution = (640, 480)
camera.rotation = 0

image = io.BytesIO()
camera.capture(image, 'jpeg')
image.seek(0)

# Lưu ảnh lại để xử lý tiếp
with open('image.jpg', 'wb') as image_file:
    image_file.write(image.read())

# Cấu hình dịch vụ Custom Vision
prediction_url = 'https://southeastasia.api.cognitive.microsoft.com/customvision/v3.0/Prediction/ed052184-4478-4262-9ffa-1a35c15c2946/detect/iterations/Iteration2/image'
prediction_key = '2a10664ae7d944ce9cba407aa2fe7b9b'

# Phân tích URL để lấy thông tin endpoint, project ID, iteration
parts = prediction_url.split('/')
endpoint = 'https://' + parts[2]
project_id = parts[6]
iteration_name = parts[9]

# Khởi tạo client Custom Vision
prediction_credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
predictor = CustomVisionPredictionClient(endpoint, prediction_credentials)

# Gửi ảnh đi để detect
image.seek(0)
results = predictor.detect_image(project_id, iteration_name, image)

# Lọc kết quả dựa trên độ chính xác
threshold = 0.3
predictions = [p for p in results.predictions if p.probability > threshold]

# Hiển thị từng prediction
for prediction in predictions:
    print(f'{prediction.tag_name}:\t{prediction.probability * 100:.2f}%')

# Đếm số lượng từng loại hàng hóa
count_by_tag = {}
for prediction in predictions:
    tag = prediction.tag_name
    count_by_tag[tag] = count_by_tag.get(tag, 0) + 1

print("\nSố lượng từng loại mặt hàng:")
for tag, count in count_by_tag.items():
    print(f'{tag}: {count} cái')

print(f'\nTổng số mặt hàng (tính từng cái riêng biệt): {len(predictions)}')

# Vẽ bounding box lên ảnh
with Image.open('image.jpg') as im:
    draw = ImageDraw.Draw(im)

    for prediction in predictions:
        scale_left = prediction.bounding_box.left
        scale_top = prediction.bounding_box.top
        scale_right = prediction.bounding_box.left + prediction.bounding_box.width
        scale_bottom = prediction.bounding_box.top + prediction.bounding_box.height
        
        left = scale_left * im.width
        top = scale_top * im.height
        right = scale_right * im.width
        bottom = scale_bottom * im.height

        draw.rectangle([left, top, right, bottom], outline=ImageColor.getrgb('red'), width=2)
        draw.text((left, top - 10), prediction.tag_name, fill=ImageColor.getrgb('red'))

    im.save('image.jpg')
