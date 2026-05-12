# # onnxruntime==1.26.0

# import onnxruntime as ort
# import numpy as np
# from PIL import Image
# import io, base64

# def preprocess(image_bytes, img_size, mean, std):
#     img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
#     img = img.resize((img_size, img_size))
#     arr = np.array(img).astype(np.float32) / 255.0
#     arr = (arr - np.array(mean)) / np.array(std)
#     return arr.transpose(2, 0, 1)[np.newaxis]  # NCHW

# def start_recognition(image_blob, model_path, top_k, useCpu):
#     if isinstance(image_blob, str):
#         if ',' in image_blob:
#             image_blob = image_blob.split(',')[1]
#         image_bytes = base64.b64decode(image_blob)
#     else:
#         image_bytes = image_blob

#     providers = ['CPUExecutionProvider'] if useCpu else [
#         'CUDAExecutionProvider', 'CPUExecutionProvider'
#     ]
#     session = ort.InferenceSession(model_path, providers=providers)

#     # mean/std/img_size нужно хранить отдельно (например, в JSON рядом с .onnx)
#     tensor = preprocess(image_bytes, img_size=224, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])

#     outputs = session.run(["output"], {"input": tensor})[0]
#     probs = softmax(outputs[0])
#     top_indices = np.argsort(probs)[::-1][:top_k]

#     return [(idx_to_label[i], float(probs[i])) for i in top_indices]

# def softmax(x):
#     e = np.exp(x - np.max(x))
#     return e / e.sum()