from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

app = FastAPI()

class PatchEmbedding(layers.Layer):
    def __init__(self, patch_size, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.projection = layers.Dense(projection_dim)
        self.layernorm = layers.LayerNormalization(epsilon=1e-6)

    def call(self, x):
        patches = tf.image.extract_patches(
            images=x,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID"
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [tf.shape(x)[0], -1, patch_dims])
        x = self.projection(patches)
        return self.layernorm(x)

model = tf.keras.models.load_model(
    "urbansound8k_transformer_model.h5",
    custom_objects={"PatchEmbedding": PatchEmbedding}
)

CLASS_NAMES = [
    'air_conditioner', 'car_horn', 'children_playing', 'dog_bark',
    'drilling', 'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music'
]

def extract_features(audio_data, sr=22050, n_mfcc=40, max_pad_len=174):
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc)
    pad_width = max_pad_len - mfccs.shape[1]
    if pad_width > 0:
        mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :max_pad_len]
    return mfccs

@app.post("/predict/")
async def predict_audio(file: UploadFile = File(...)):
    try:
        audio, sr = librosa.load(file.file, sr=22050, res_type='kaiser_fast')

        mfcc = extract_features(audio)
        X = np.expand_dims(mfcc, axis=(0, -1))

        preds = model.predict(X)
        pred_class = CLASS_NAMES[np.argmax(preds)]
        conf = np.max(preds)

        return {"Predicted Class": pred_class, "Confidence": float(conf)}

    except Exception as e:
        import traceback
        print(" ERROR during prediction:")
        traceback.print_exc()

        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
