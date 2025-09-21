# =========================
# AI-EcoScan – Hugging Face Streamlit App
# =========================

import streamlit as st
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch, wikipedia, matplotlib.pyplot as plt
import numpy as np
import requests
from io import BytesIO

# ------------------------
# 1. Load CLIP for material detection
# ------------------------
@st.cache_resource
def load_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

model, processor = load_model()

candidate_labels = [
    "steel", "aluminum", "plastic", "glass", "copper", "wood",
    "rubber", "concrete", "paper", "ceramic", "carbon fiber"
]

# ------------------------
# 2. Local knowledge base
# ------------------------
sustainability_db = {
    "steel": {
        "impact": "High CO₂ emissions from smelting, but recyclable.",
        "alternatives": ["Recycled steel", "Aluminum"],
        "mechanical": {"strength": 9, "ductility": 6, "density": 9, "sustainability": 3},
        "adaptation": ["Powder metallurgy", "Heat treatment"]
    },
    "plastic": {
        "impact": "Derived from petroleum, very high persistence in environment.",
        "alternatives": ["Bioplastics", "Bamboo composites"],
        "mechanical": {"strength": 3, "ductility": 8, "density": 2, "sustainability": 2},
        "adaptation": ["Fiber reinforcement", "Surface coating"]
    },
    "aluminum": {
        "impact": "Energy-intensive to produce, but lightweight and recyclable.",
        "alternatives": ["Recycled aluminum", "Magnesium alloys"],
        "mechanical": {"strength": 6, "ductility": 8, "density": 3, "sustainability": 7},
        "adaptation": ["Alloying", "Heat treatment"]
    }
}

# ------------------------
# 3. Classification
# ------------------------
def classify_material(image):
    inputs = processor(text=candidate_labels, images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)
    pred = candidate_labels[probs.argmax().item()]
    return pred

# ------------------------
# 4. Online fallback
# ------------------------
def fetch_from_wikipedia(material, query_type="impact"):
    try:
        if query_type == "impact":
            query = f"Environmental impact of {material}"
        elif query_type == "mechanical":
            query = f"Mechanical properties of {material}"
        else:
            query = material
        return wikipedia.summary(query, sentences=3)
    except:
        return "No reliable online data found."

def analyze_material(material):
    if material in sustainability_db:
        return sustainability_db[material]
    else:
        return {
            "impact": fetch_from_wikipedia(material, "impact"),
            "alternatives": [f"Recycled {material}", f"Eco-friendly {material}"],
            "mechanical": {"strength": 5, "ductility": 5, "density": 5, "sustainability": 5},
            "adaptation": ["Check sustainable processing methods"]
        }

def analyze_alternative(alt):
    alt_key = alt.lower().replace(" ", "")
    match = [k for k in sustainability_db.keys() if k.lower().replace(" ", "") == alt_key]
    if match:
        return sustainability_db[match[0]]
    else:
        return {
            "impact": fetch_from_wikipedia(alt, "impact"),
            "mechanical": {"strength": 5, "ductility": 5, "density": 5, "sustainability": 6},
            "adaptation": ["Look into green adaptations"],
            "alternatives": []
        }

# ------------------------
# 5. Comparison
# ------------------------
def compare_materials(material, data):
    original = data["mechanical"]

    if not data["alternatives"]:
        return "No alternatives found."

    best_alt = data["alternatives"][0]
    alt_data = analyze_alternative(best_alt)
    alt_mech = alt_data["mechanical"]

    verdict = f"### Comparison: {material.capitalize()} vs {best_alt}\n\n"
    if alt_mech["sustainability"] > original["sustainability"]:
        if alt_mech["strength"] >= original["strength"] - 2:
            verdict += f"✅ {best_alt} is more eco-friendly and maintains comparable strength. **Recommended.**"
        else:
            verdict += f"⚠️ {best_alt} is more eco-friendly but has weaker strength. Possible trade-off."
    else:
        verdict += f"❌ {best_alt} does not provide a better balance of sustainability and performance."

    verdict += f"\n\n**Environmental Info:**\n- {material.capitalize()}: {data['impact']}\n- {best_alt}: {alt_data['impact']}\n"

    return verdict

# ------------------------
# 6. Radar Chart
# ------------------------
def plot_comparison(material, data):
    categories = ["strength", "ductility", "density", "sustainability"]
    N = len(categories)

    orig_values = [data["mechanical"].get(cat, 5) for cat in categories]

    alt_name, alt_values = None, None
    if data["alternatives"]:
        alt = data["alternatives"][0]
        alt_data = analyze_alternative(alt)
        alt_name = alt
        alt_values = [alt_data["mechanical"].get(cat, 5) for cat in categories]

    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    values = orig_values + [orig_values[0]]
    angles += [angles[0]]

    fig, ax = plt.subplots(figsize=(5,5), subplot_kw=dict(polar=True))
    ax.plot(angles, values, "o-", linewidth=2, label=material.capitalize())
    ax.fill(angles, values, alpha=0.25)

    if alt_values:
        alt_values = alt_values + [alt_values[0]]
        ax.plot(angles, alt_values, "o-", linewidth=2, label=alt_name)
        ax.fill(angles, alt_values, alpha=0.25)

    ax.set_thetagrids(np.degrees(angles[:-1]), categories)
    ax.set_ylim(0, 10)
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))
    return fig

# ------------------------
# 7. Streamlit UI
# ------------------------
st.title(" AI-EcoScan – Eco Materials Advisor ")
st.markdown("Upload a material image. The system will detect, analyze, and compare its environmental and mechanical performance with alternatives.")

uploaded_file = st.file_uploader("Upload / Capture Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Analyzing material..."):
        material = classify_material(img)
        data = analyze_material(material)

        st.subheader("AI-EcoScan Report")
        st.write(f"**Detected Material:** {material.capitalize()}")
        st.write(f"**Environmental Impact:** {data['impact']}")
        st.write(f"**Possible Alternatives:** {', '.join(data['alternatives']) if data['alternatives'] else 'None found'}")
        st.write(f"**Adaptation Techniques:** {', '.join(data['adaptation'])}")

        recommendation = compare_materials(material, data)
        st.markdown(recommendation)

        fig = plot_comparison(material, data)
        st.pyplot(fig)
