import streamlit as st
import cv2
import numpy as np
import zipfile
import tempfile
import os
from PIL import Image
from insightface.app import FaceAnalysis

st.set_page_config(page_title="Face Search", layout="wide")

# Load model once
@st.cache_resource
def load_model():
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0)
    return app

app = load_model()

def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

st.title("🔍 Face Search")

reference = st.file_uploader(
    "Upload reference image",
    type=["jpg", "jpeg", "png"]
)

zip_file = st.file_uploader(
    "Upload ZIP containing photos",
    type=["zip"]
)

threshold = st.slider("Similarity Threshold", 0.3, 0.9, 0.6)

if st.button("Search"):

    if reference is None or zip_file is None:
        st.warning("Upload both files first.")
        st.stop()

    ref = np.array(Image.open(reference).convert("RGB"))
    ref = cv2.cvtColor(ref, cv2.COLOR_RGB2BGR)

    faces = app.get(ref)

    if len(faces) == 0:
        st.error("No face found in reference image.")
        st.stop()

    ref_embedding = faces[0].embedding

    matches = []

    with tempfile.TemporaryDirectory() as temp:

        with zipfile.ZipFile(zip_file, "r") as z:
            z.extractall(temp)

        for root, _, files in os.walk(temp):

            for file in files:

                if not file.lower().endswith(
                    (".jpg", ".jpeg", ".png")
                ):
                    continue

                path = os.path.join(root, file)

                img = cv2.imread(path)

                if img is None:
                    continue

                detected = app.get(img)

                found = False

                for face in detected:

                    sim = cosine(ref_embedding, face.embedding)

                    if sim >= threshold:

                        x1, y1, x2, y2 = map(int, face.bbox)

                        cv2.rectangle(
                            img,
                            (x1, y1),
                            (x2, y2),
                            (0,255,0),
                            3
                        )

                        cv2.putText(
                            img,
                            f"{sim:.2f}",
                            (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0,255,0),
                            2
                        )

                        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                        matches.append((file, rgb, sim))
                        found = True
                        break

    st.success(f"{len(matches)} matching images found.")

    cols = st.columns(3)

    for i, (name, image, sim) in enumerate(matches):

        with cols[i % 3]:
            st.image(
                image,
                caption=f"{name}\nSimilarity: {sim:.2f}",
                use_container_width=True
            )
