import base64
import threading

import cv2
import numpy as np

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi import WebSocketDisconnect

from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model


app = FastAPI(
    title="Neon AI Face Swap"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


MODEL_NAME = "inswapper_128.onnx"

face_app = None
swapper = None

model_lock = threading.Lock()


def init_models():

    global face_app
    global swapper

    if face_app is not None:
        return

    print("AI modelleri hazırlanıyor...")

    face_app = FaceAnalysis(
        name="buffalo_l",
        providers=[
            "CUDAExecutionProvider",
            "CPUExecutionProvider"
        ]
    )

    face_app.prepare(
        ctx_id=-1,
        det_size=(640, 640)
    )

    swapper = get_model(
        MODEL_NAME,
        providers=[
            "CUDAExecutionProvider",
            "CPUExecutionProvider"
        ]
    )

    print("AI modelleri hazır.")


def decode_image(data_url):

    if "," in data_url:
        data_url = data_url.split(",", 1)[1]

    raw = base64.b64decode(data_url)

    array = np.frombuffer(
        raw,
        dtype=np.uint8
    )

    image = cv2.imdecode(
        array,
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise ValueError(
            "Görüntü çözülemedi."
        )

    return image


def encode_image(image):

    success, encoded = cv2.imencode(
        ".jpg",
        image,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            82
        ]
    )

    if not success:
        raise ValueError(
            "Görüntü JPEG olarak kodlanamadı."
        )

    return (
        "data:image/jpeg;base64,"
        + base64.b64encode(
            encoded.tobytes()
        ).decode("ascii")
    )


def find_largest_face(image):

    faces = face_app.get(image)

    if not faces:
        raise ValueError(
            "Kaynak fotoğrafta yüz bulunamadı."
        )

    faces.sort(
        key=lambda face:
        (face.bbox[2] - face.bbox[0])
        *
        (face.bbox[3] - face.bbox[1]),
        reverse=True
    )

    return faces[0]


def swap_frame(
    image,
    source_face
):

    faces = face_app.get(image)

    if not faces:
        return image

    output = image.copy()

    # İlk ve en büyük yüz
    target = max(
        faces,
        key=lambda face:
        (face.bbox[2] - face.bbox[0])
        *
        (face.bbox[3] - face.bbox[1])
    )

    output = swapper.get(
        output,
        target,
        source_face,
        paste_back=True
    )

    return output


@app.get("/health")
def health():

    init_models()

    return {
        "ok": True,
        "model": MODEL_NAME
    }


@app.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket
):

    await websocket.accept()

    source_face = None

    try:

        init_models()

        await websocket.send_json({
            "type": "ready",
            "model": MODEL_NAME
        })

        while True:

            message = (
                await websocket.receive_json()
            )

            message_type = \
                message.get("type")

            if message_type == "source":

                source_image = \
                    decode_image(
                        message["image"]
                    )

                with model_lock:

                    source_face = \
                        find_largest_face(
                            source_image
                        )

                await websocket.send_json({
                    "type": "source-ready"
                })

            elif message_type == "frame":

                if source_face is None:

                    await websocket.send_json({
                        "type": "error",
                        "message":
                        "Önce kaynak yüzünü yükle."
                    })

                    continue

                frame = decode_image(
                    message["image"]
                )

                with model_lock:

                    result = swap_frame(
                        frame,
                        source_face
                    )

                encoded = encode_image(
                    result
                )

                await websocket.send_json({
                    "type": "frame",
                    "image": encoded
                })

    except WebSocketDisconnect:

        print(
            "WebSocket bağlantısı kapandı."
        )

    except Exception as error:

        print(
            "Sunucu hatası:",
            repr(error)
        )

        try:

            await websocket.send_json({
                "type": "error",
                "message": str(error)
            })

        except Exception:
            pass


if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )
