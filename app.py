# =========================
# AI-EcoScan – Streamlit Version
# =========================

import streamlit as st
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import wikipedia
import matplotlib.pyplot as plt
import numpy as np
from googlesearch import search

# ------------------------
# 1. Load CLIP model
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
def fetch_from_google(query, num_results=2):
    try:
        urls = list(search(query, num_results=num_results))
        return "\n".join(urls) if urls else "No results found."
    except:
        return "Google search failed."

def fetch_info(material, query_type="impact"):
    try:
        if query_type == "impact":
            return wikipedia.summary(f"Environmental impact of {material}", sentences=3)
        elif query_type == "mechanical":
            return wikipedia.summary(f"Mechanical properties of {material}", sentences=3)
        else:
            return wikipedia.summary(material, sentences=3)
    except:
        return fetch_from_google(f"{material} {query_type}")

def analyze_material(material):
    if material in sustainability_db:
        return sustainability_db[material]
    else:
        return {
            "impact": fetch_info(material, "impact"),
            "alternatives": ["Recycled " + material, "Eco-friendly " + material],
            "mechanical": {"strength": 5, "ductility": 5, "density": 5, "sustainability": 5},
            "adaptation": ["Search sustainable processing methods"]
        }

def analyze_alternative(alt):
    key = alt.lower().replace(" ", "")
    match = [k for k in sustainability_db.keys() if k.lower().replace(" ", "") == key]
    if match:
        return sustainability_db[match[0]]
    else:
        return {
            "impact": fetch_info(alt, "impact"),
            "mechanical": {"strength": 5, "ductility": 5, "density": 5, "sustainability": 6},
            "adaptation": ["Look into green adaptations"],
            "alternatives": []
        }

# ------------------------
# 5. Comparison & Radar Chart
# ------------------------
def compare_materials(material, data):
    if not data["alternatives"]:
        return "No alternatives found."
    best_alt = data["alternatives"][0]
    alt_data = analyze_alternative(best_alt)
    verdict = f"Comparison: {material.capitalize()} vs {best_alt}\n\n"
    if alt_data["mechanical"]["sustainability"] > data["mechanical"]["sustainability"]:
        verdict += f"{best_alt} is more eco-friendly."
    else:
        verdict += f"{best_alt} does not improve sustainability."
    verdict += f"\n\nEnvironmental Info:\n- {material.capitalize()}: {data['impact']}\n- {best_alt}: {alt_data['impact']}\n"
    return verdict

def plot_radar(material, data):
    categories = ["strength", "ductility", "density", "sustainability"]
    N = len(categories)
    orig_values = [data["mechanical"].get(cat, 5) for cat in categories]

    alt_values = None
    alt_name = None
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
# 6. Streamlit App
# ------------------------
st.set_page_config(page_title="AI-EcoScan", layout="wide")
st.markdown("<h1 style='text-align:center; color: green;'>AI-EcoScan – Eco Materials Advisor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Upload a material image. The system will detect, analyze, and compare its environmental and mechanical performance.</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload / Capture Image", type=["png","jpg","jpeg"])
if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    material = classify_material(img)
    data = analyze_material(material)

    st.subheader("AI-EcoScan Report")
    st.text(f"Detected Material: {material.capitalize()}")
    st.text(f"Environmental Impact:\n{data['impact']}")
    st.text(f"Possible Alternatives: {', '.join(data['alternatives']) if data['alternatives'] else 'None found'}")
    st.text(f"Adaptation Techniques: {', '.join(data['adaptation'])}")

    st.subheader("Recommendation")
    st.text(compare_materials(material, data))

    st.subheader("Mechanical Comparison Radar Chart")
    st.pyplot(plot_radar(material, data))
