# ← TOP OF FILE — before everything else
import gdown
import os

os.makedirs('models', exist_ok=True)

if not os.path.exists('models/plant_model.tflite'):
    print("Downloading model...")
    gdown.download(
        'https://drive.google.com/uc?id=10VDRbly9Y-DyN4HZZvHhk1YGRjT1BF6Q',
        'models/plant_model.tflite',
        quiet=False
    )
    print("✅ Model downloaded!")

if not os.path.exists('models/class_names.json'):
    print("Downloading class names...")
    gdown.download(
        'https://drive.google.com/uc?id=1zwSE4wJ0dARnoid9Njz_0SDOBpGbm1Sd',
        'models/class_names.json',
        quiet=False
    )
    print("✅ Class names downloaded!")

# ← REST OF YOUR APP CODE BELOW
import streamlit as st
import numpy as np
import streamlit as st
import numpy as np
from PIL import Image
import json
import os
import time

# ── TFLite interpreter — no Keras, no quantization error ──
import tensorflow as tf

# ─────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CropCare AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,600;0,700;1,600&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ── Background ── */
.stApp {
    background: linear-gradient(160deg, #0d1f17 0%, #1a3c2e 40%, #0d1f17 100%);
    min-height: 100vh;
}

/* ── Hero ── */
.hero-block {
    background: linear-gradient(135deg, rgba(45,106,79,0.6), rgba(82,183,136,0.2));
    border: 1px solid rgba(82,183,136,0.25);
    border-radius: 20px;
    padding: 2.5rem 2rem 2rem;
    margin-bottom: 2rem;
    text-align: center;
    backdrop-filter: blur(10px);
}
.hero-badge {
    display: inline-block;
    background: rgba(82,183,136,0.15);
    border: 1px solid rgba(82,183,136,0.3);
    color: #52b788;
    padding: 4px 16px;
    border-radius: 50px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    margin-bottom: 1rem;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.6rem;
    color: #ffffff;
    margin: 0 0 0.5rem;
    line-height: 1.15;
}
.hero-title em { color: #52b788; font-style: italic; }
.hero-sub {
    color: rgba(255,255,255,0.65);
    font-size: 1rem;
    font-weight: 300;
    margin: 0;
}
.crop-pills {
    margin-top: 1.2rem;
    display: flex;
    gap: 0.6rem;
    justify-content: center;
    flex-wrap: wrap;
}
.crop-pill {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15);
    color: rgba(255,255,255,0.8);
    padding: 4px 14px;
    border-radius: 50px;
    font-size: 0.82rem;
}

/* ── Cards ── */
.glass-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1.2rem;
    backdrop-filter: blur(8px);
}

/* ── Result Cards ── */
.result-healthy {
    background: rgba(67,160,71,0.12);
    border: 1px solid rgba(67,160,71,0.35);
    border-radius: 16px;
    padding: 1.5rem;
    margin: 1rem 0;
}
.result-warning {
    background: rgba(251,140,0,0.12);
    border: 1px solid rgba(251,140,0,0.35);
    border-radius: 16px;
    padding: 1.5rem;
    margin: 1rem 0;
}
.result-danger {
    background: rgba(229,57,53,0.12);
    border: 1px solid rgba(229,57,53,0.35);
    border-radius: 16px;
    padding: 1.5rem;
    margin: 1rem 0;
}

.disease-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.6rem;
    color: white;
    margin: 0.3rem 0;
}
.confidence-text {
    color: rgba(255,255,255,0.7);
    font-size: 0.9rem;
    margin: 0;
}

/* ── Info Grid ── */
.info-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.8rem;
    margin-top: 1rem;
}
.info-item {
    background: rgba(255,255,255,0.05);
    border-radius: 10px;
    padding: 0.8rem 1rem;
    border: 1px solid rgba(255,255,255,0.08);
}
.info-label {
    font-size: 0.72rem;
    font-weight: 700;
    color: #52b788;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.3rem;
}
.info-value {
    font-size: 0.88rem;
    color: rgba(255,255,255,0.8);
    line-height: 1.4;
}

/* ── Top 3 ── */
.top3-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.5rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.07);
    color: rgba(255,255,255,0.75);
    font-size: 0.88rem;
}
.top3-pct {
    color: #52b788;
    font-weight: 600;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: rgba(13,31,23,0.95) !important;
    border-right: 1px solid rgba(82,183,136,0.15);
}
section[data-testid="stSidebar"] * {
    color: rgba(255,255,255,0.85) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #2d6a4f, #52b788) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.65rem 2rem !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    width: 100% !important;
    transition: all 0.3s !important;
    letter-spacing: 0.02em !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(82,183,136,0.35) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.04) !important;
    border: 2px dashed rgba(82,183,136,0.4) !important;
    border-radius: 12px !important;
}

/* ── Metric ── */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.05);
    border-radius: 10px;
    padding: 0.8rem;
    border: 1px solid rgba(255,255,255,0.08);
}
[data-testid="stMetricLabel"] { color: rgba(255,255,255,0.6) !important; }
[data-testid="stMetricValue"] { color: white !important; }

/* ── Progress bar ── */
.stProgress > div > div {
    background: linear-gradient(90deg, #2d6a4f, #52b788) !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab"] {
    color: rgba(255,255,255,0.6) !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stTabs [aria-selected="true"] {
    color: #52b788 !important;
    border-bottom-color: #52b788 !important;
}

/* ── General text ── */
p, li, span { color: rgba(255,255,255,0.8); }
h1, h2, h3  { color: white; font-family: 'Playfair Display', serif; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# DISEASE DATABASE
# ─────────────────────────────────────────────────────────
DISEASE_INFO = {
    'Tomato___Early_blight': {
        'cause': 'Fungal (Alternaria solani)',
        'treatment': 'Apply copper-based fungicide every 7 days',
        'prevention': 'Avoid overhead watering, rotate crops yearly',
        'urgency': 'Act within 3-5 days',
        'level': 'warning'
    },
    'Tomato___Late_blight': {
        'cause': 'Oomycete (Phytophthora infestans)',
        'treatment': 'Apply mancozeb or chlorothalonil fungicide',
        'prevention': 'Plant resistant varieties, improve air circulation',
        'urgency': 'Act immediately',
        'level': 'danger'
    },
    'Tomato___healthy': {
        'cause': 'No disease detected',
        'treatment': 'No treatment needed',
        'prevention': 'Maintain regular watering and fertilization',
        'urgency': 'Crop is healthy',
        'level': 'healthy'
    },
    'Tomato___Leaf_Mold': {
        'cause': 'Fungal (Passalora fulva)',
        'treatment': 'Apply fungicide, improve ventilation',
        'prevention': 'Reduce humidity, avoid wetting leaves',
        'urgency': 'Act within 3-5 days',
        'level': 'warning'
    },
    'Tomato___Septoria_leaf_spot': {
        'cause': 'Fungal (Septoria lycopersici)',
        'treatment': 'Apply chlorothalonil-based fungicide',
        'prevention': 'Avoid overhead irrigation, remove infected leaves',
        'urgency': 'Act within 3-5 days',
        'level': 'warning'
    },
    'Tomato___Spider_mites Two-spotted_spider_mite': {
        'cause': 'Spider mite (Tetranychus urticae)',
        'treatment': 'Apply miticide or neem oil spray',
        'prevention': 'Maintain humidity, avoid water stress',
        'urgency': 'Act within 2-3 days',
        'level': 'warning'
    },
    'Tomato___Target_Spot': {
        'cause': 'Fungal (Corynespora cassiicola)',
        'treatment': 'Apply azoxystrobin-based fungicide',
        'prevention': 'Improve air circulation, avoid leaf wetness',
        'urgency': 'Act within 3-5 days',
        'level': 'warning'
    },
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {
        'cause': 'Viral — spread by whiteflies',
        'treatment': 'Remove infected plants, control whiteflies',
        'prevention': 'Use virus-resistant varieties, insect nets',
        'urgency': 'Act immediately',
        'level': 'danger'
    },
    'Tomato___Tomato_mosaic_virus': {
        'cause': 'Tomato mosaic virus (ToMV)',
        'treatment': 'Remove infected plants immediately',
        'prevention': 'Use certified virus-free seeds, sanitize tools',
        'urgency': 'Act immediately',
        'level': 'danger'
    },
    'Tomato___Bacterial_spot': {
        'cause': 'Bacterial (Xanthomonas spp.)',
        'treatment': 'Apply copper-based bactericide',
        'prevention': 'Use disease-free seeds, avoid wet field work',
        'urgency': 'Act within 2-3 days',
        'level': 'warning'
    },
    'Potato___Early_blight': {
        'cause': 'Fungal (Alternaria solani)',
        'treatment': 'Apply chlorothalonil-based fungicide',
        'prevention': 'Use certified seed potatoes, avoid wet conditions',
        'urgency': 'Act within 3-5 days',
        'level': 'warning'
    },
    'Potato___Late_blight': {
        'cause': 'Oomycete (Phytophthora infestans)',
        'treatment': 'Apply metalaxyl-based fungicide immediately',
        'prevention': 'Use resistant varieties, avoid excess irrigation',
        'urgency': 'Act immediately',
        'level': 'danger'
    },
    'Potato___healthy': {
        'cause': 'No disease detected',
        'treatment': 'No treatment needed',
        'prevention': 'Maintain proper soil drainage and fertilization',
        'urgency': 'Crop is healthy',
        'level': 'healthy'
    },
    'Pepper__bell___Bacterial_spot': {
        'cause': 'Bacterial (Xanthomonas campestris)',
        'treatment': 'Apply copper-based bactericide',
        'prevention': 'Use disease-free seeds, avoid wet field work',
        'urgency': 'Act within 2-3 days',
        'level': 'warning'
    },
    'Pepper__bell___healthy': {
        'cause': 'No disease detected',
        'treatment': 'No treatment needed',
        'prevention': 'Maintain regular watering and fertilization',
        'urgency': 'Crop is healthy',
        'level': 'healthy'
    }
}

TRANSLATIONS = {
    'en': {
        'title': 'Detect Plant Disease <em>Instantly</em>',
        'sub': 'Upload a leaf image and get AI-powered diagnosis in seconds',
        'upload_label': 'Upload Leaf Image',
        'analyse': '🔍 Analyse Disease',
        'detected': 'Detected Disease',
        'confidence': 'Confidence',
        'cause': 'Cause',
        'treatment': 'Treatment',
        'prevention': 'Prevention',
        'urgency': 'Urgency',
        'top3': 'Top 3 Predictions',
        'details': 'Image Details',
        'analysing': 'Analysing your crop...',
        'healthy_msg': '✅ Your crop looks healthy! Keep monitoring regularly.',
        'footer': 'CropCare AI · Powered by MobileNetV2 + TFLite · PlantVillage Dataset'
    },
    'hi': {
        'title': 'पौधे की बीमारी पहचानें <em>तुरंत</em>',
        'sub': 'पत्ती की छवि अपलोड करें और AI से निदान पाएं',
        'upload_label': 'पत्ती की छवि अपलोड करें',
        'analyse': '🔍 बीमारी पहचानें',
        'detected': 'पहचानी गई बीमारी',
        'confidence': 'विश्वास',
        'cause': 'कारण',
        'treatment': 'उपचार',
        'prevention': 'बचाव',
        'urgency': 'तात्कालिकता',
        'top3': 'शीर्ष 3 भविष्यवाणियाँ',
        'details': 'छवि विवरण',
        'analysing': 'आपकी फसल का विश्लेषण हो रहा है...',
        'healthy_msg': '✅ आपकी फसल स्वस्थ दिखती है! नियमित रूप से निगरानी रखें।',
        'footer': 'CropCare AI · MobileNetV2 + TFLite द्वारा संचालित'
    },
    'mr': {
        'title': 'वनस्पती रोग ओळखा <em>त्वरित</em>',
        'sub': 'पानाची प्रतिमा अपलोड करा आणि AI निदान मिळवा',
        'upload_label': 'पानाची प्रतिमा अपलोड करा',
        'analyse': '🔍 रोग ओळखा',
        'detected': 'ओळखलेला रोग',
        'confidence': 'विश्वास',
        'cause': 'कारण',
        'treatment': 'उपचार',
        'prevention': 'प्रतिबंध',
        'urgency': 'निकड',
        'top3': 'शीर्ष 3 अंदाज',
        'details': 'प्रतिमा तपशील',
        'analysing': 'आपल्या पिकाचे विश्लेषण होत आहे...',
        'healthy_msg': '✅ आपले पीक निरोगी दिसते! नियमित देखरेख ठेवा.',
        'footer': 'CropCare AI · MobileNetV2 + TFLite द्वारे समर्थित'
    }
}

# ─────────────────────────────────────────────────────────
# LOAD TFLITE MODEL — No Keras, No quantization error
# ─────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_tflite_model():
    model_path      = os.path.join(BASE_DIR, 'models', 'plant_model.tflite')
    class_path      = os.path.join(BASE_DIR, 'models', 'class_names.json')

    # Load TFLite interpreter
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Load class names
    with open(class_path, 'r') as f:
        class_names = json.load(f)

    return interpreter, class_names

def predict_tflite(interpreter, class_names, img_array):
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], img_array.astype(np.float32))
    interpreter.invoke()

    # Get output
    predictions = interpreter.get_tensor(output_details[0]['index'])[0]

    # Top 3
    top3_indices = np.argsort(predictions)[::-1][:3]
    top3 = [
        {'disease': class_names[str(i)], 'confidence': float(predictions[i] * 100)}
        for i in top3_indices
    ]

    predicted_index = top3_indices[0]
    disease         = class_names[str(predicted_index)]
    confidence      = float(predictions[predicted_index] * 100)

    return disease, confidence, top3

def preprocess(image: Image.Image):
    image     = image.convert('RGB').resize((224, 224))
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ─────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🌿 CropCare AI")
    st.markdown("---")

    lang = st.selectbox(
        "🌍 Language",
        options=['en', 'hi', 'mr'],
        format_func=lambda x: {'en': '🇬🇧 English', 'hi': '🇮🇳 Hindi', 'mr': '🇮🇳 Marathi'}[x]
    )
    t = TRANSLATIONS[lang]

    st.markdown("---")
    st.markdown("**🌾 Supported Crops**")
    st.markdown("🍅 Tomato &nbsp; 🥔 Potato &nbsp; 🫑 Pepper")

    st.markdown("---")
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

    st.markdown("---")
    st.markdown("**📌 Tips for best results**")
    st.caption("• Use clear, well-lit images")
    st.caption("• Single leaf per photo")
    st.caption("• Avoid blurry images")
    st.caption("• Natural light works best")

    st.markdown("---")
    # Voice output button
    if st.button("🔊 Speak Result"):
        st.session_state['speak'] = True

# ─────────────────────────────────────────────────────────
# MAIN UI
# ─────────────────────────────────────────────────────────

# Hero
st.markdown(f"""
<div class="hero-block">
    <div class="hero-badge">AI Powered • Free • Instant</div>
    <h1 class="hero-title">🌿 {t['title']}</h1>
    <p class="hero-sub">{t['sub']}</p>
    <div class="crop-pills">
        <span class="crop-pill">🍅 Tomato</span>
        <span class="crop-pill">🥔 Potato</span>
        <span class="crop-pill">🫑 Pepper</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Load model
try:
    interpreter, class_names = load_tflite_model()
except Exception as e:
    st.error(f"❌ Model loading failed: {e}")
    st.info("Make sure `models/plant_model.tflite` and `models/class_names.json` are in the app folder.")
    st.stop()

# Upload
uploaded_file = st.file_uploader(
    t['upload_label'],
    type=['jpg', 'jpeg', 'png', 'webp']
)

if uploaded_file:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.image(image, caption="Uploaded Leaf", use_column_width=True)

        st.markdown(f"""
        <div class="glass-card">
            <div style="color:#52b788; font-weight:700; font-size:0.8rem; 
                        text-transform:uppercase; letter-spacing:0.08em; 
                        margin-bottom:0.8rem">{t['details']}</div>
            <div class="top3-item"><span>Format</span><span>{image.format or 'N/A'}</span></div>
            <div class="top3-item"><span>Size</span><span>{image.width} × {image.height} px</span></div>
            <div class="top3-item"><span>Mode</span><span>{image.mode}</span></div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        if st.button(t['analyse']):
            with st.spinner(t['analysing']):
                time.sleep(0.5)
                img_array           = preprocess(image)
                disease, confidence, top3 = predict_tflite(interpreter, class_names, img_array)

            # Store in session for voice
            st.session_state['last_disease']    = disease
            st.session_state['last_confidence'] = confidence
            st.session_state['last_info']       = DISEASE_INFO.get(disease, {})

            # Get info
            info  = DISEASE_INFO.get(disease, {
                'cause': 'Not available',
                'treatment': 'Consult an agricultural expert',
                'prevention': 'Maintain good crop hygiene',
                'urgency': 'Consult an expert',
                'level': 'warning'
            })
            level = info.get('level', 'warning')

            # Disease card
            card_class = f"result-{level}"
            icon = '✅' if level == 'healthy' else ('🚨' if level == 'danger' else '⚠️')
            display_name = disease.replace('___', ' — ').replace('_', ' ')

            st.markdown(f"""
            <div class="{card_class}">
                <div style="font-size:0.75rem; font-weight:700; color:rgba(255,255,255,0.5);
                            text-transform:uppercase; letter-spacing:0.1em">{t['detected']}</div>
                <div class="disease-title">{icon} {display_name}</div>
                <div class="confidence-text">{t['confidence']}: <strong>{confidence:.1f}%</strong></div>
            </div>
            """, unsafe_allow_html=True)

            # Confidence warning
            if confidence < confidence_threshold * 100 and level != 'healthy':
                st.warning(f"⚠️ Confidence ({confidence:.1f}%) is below your threshold ({confidence_threshold*100:.0f}%). Consider consulting an agronomist.")

            # Healthy message
            if level == 'healthy':
                st.success(t['healthy_msg'])

            # Confidence bar
            st.progress(int(confidence))

            # Info grid
            st.markdown(f"""
            <div class="info-grid">
                <div class="info-item">
                    <div class="info-label">🦠 {t['cause']}</div>
                    <div class="info-value">{info['cause']}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">💊 {t['treatment']}</div>
                    <div class="info-value">{info['treatment']}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">🛡️ {t['prevention']}</div>
                    <div class="info-value">{info['prevention']}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">⏰ {t['urgency']}</div>
                    <div class="info-value">{info['urgency']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Top 3
            st.markdown(f"""
            <div class="glass-card" style="margin-top:1rem">
                <div style="color:#52b788; font-weight:700; font-size:0.8rem;
                            text-transform:uppercase; letter-spacing:0.08em;
                            margin-bottom:0.8rem">{t['top3']}</div>
            """, unsafe_allow_html=True)

            for i, item in enumerate(top3):
                name = item['disease'].replace('___', ' — ').replace('_', ' ')
                pct  = item['confidence']
                medal = ['🥇', '🥈', '🥉'][i]
                st.markdown(f"""
                <div class="top3-item">
                    <span>{medal} {name}</span>
                    <span class="top3-pct">{pct:.1f}%</span>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

            # Voice JS injection
            voice_text = f"{display_name}. Confidence {confidence:.0f} percent. {info['urgency']}. {info['treatment']}"
            lang_code  = {'en': 'en-US', 'hi': 'hi-IN', 'mr': 'mr-IN'}[lang]
            st.components.v1.html(f"""
            <script>
                var msg = new SpeechSynthesisUtterance("{voice_text}");
                msg.lang  = "{lang_code}";
                msg.rate  = 0.88;
                window.speechSynthesis.cancel();
                window.speechSynthesis.speak(msg);
            </script>
            """, height=0)

else:
    st.markdown("""
    <div class="glass-card" style="text-align:center; padding:3rem">
        <div style="font-size:3rem; margin-bottom:1rem">📸</div>
        <div style="color:rgba(255,255,255,0.6); font-size:1rem">
            Upload a leaf image above to get started
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(f"<p style='text-align:center; color:rgba(255,255,255,0.35); font-size:0.8rem'>{t['footer']}</p>",
            unsafe_allow_html=True)
