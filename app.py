
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode


# ==========================================
# Load Clinical AI
# ==========================================

clinical_model = joblib.load(
     "ocuneuro_clinical_model.pkl"
)

clinical_features = joblib.load(
     "ocuneuro_clinical_features.pkl"
)

clinical_threshold = joblib.load(
     "ocuneuro_clinical_threshold.pkl"
)


# ==========================================
# U-Net
# ==========================================

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.enc1 = DoubleConv(3, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(512, 1024)

        self.up4 = nn.ConvTranspose2d(
            1024, 512, 2, stride=2
        )
        self.dec4 = DoubleConv(1024, 512)

        self.up3 = nn.ConvTranspose2d(
            512, 256, 2, stride=2
        )
        self.dec3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(
            256, 128, 2, stride=2
        )
        self.dec2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(
            128, 64, 2, stride=2
        )
        self.dec1 = DoubleConv(128, 64)

        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, x):

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.out(d1)


# ==========================================
# Load U-Net
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

retinal_model = UNet().to(device)

checkpoint = torch.load(
    "/content/ocuneuro_unet_best.pth",
    map_location=device
)

retinal_model.load_state_dict(
    checkpoint["model_state_dict"]
)

retinal_model.eval()


# ==========================================
# Streamlit Page
# ==========================================

st.set_page_config(
    page_title="OcuNeuro OS",
    page_icon="👁️",
    layout="wide"
)

st.title("👁️ OcuNeuro OS")
st.subheader(
    "AI-assisted Stroke Risk Screening Prototype"
)

st.info(
    "ระบบต้นแบบสำหรับการคัดกรองความเสี่ยงที่เกี่ยวข้องกับ "
    "โรคหลอดเลือดสมอง ไม่ใช่เครื่องมือสำหรับการวินิจฉัยโรค"
)


# ==========================================
# Retinal AI
# ==========================================

st.header("1. Retinal AI")

uploaded_file = st.file_uploader(
    "อัปโหลดภาพจอประสาทตา (Fundus Image)",
    type=["png", "jpg", "jpeg"]
)

retinal_density = None
retinal_area = None


if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:

        st.image(
            image,
            caption="Original Fundus Image",
            width=500
        )

    # -----------------------------
    # Prepare image for U-Net
    # -----------------------------

    input_image = TF.resize(
        image,
        [512, 512],
        interpolation=InterpolationMode.BILINEAR
    )

    input_tensor = TF.to_tensor(
        input_image
    ).unsqueeze(0).to(device)


    # -----------------------------
    # U-Net Prediction
    # -----------------------------

    with torch.no_grad():

        output = retinal_model(
            input_tensor
        )

        probability = torch.sigmoid(
            output
        )

        prediction = (
            probability > 0.5
        ).float()


    vessel_mask = (
        prediction
        .squeeze()
        .cpu()
        .numpy()
    )


    # -----------------------------
    # Vessel Features
    # -----------------------------

    total_pixels = vessel_mask.size

    retinal_area = float(
        vessel_mask.sum()
    )

    retinal_density = (
        retinal_area / total_pixels
    )


    # -----------------------------
    # Display Vessel Mask
    # -----------------------------

    with col2:

        st.image(
            vessel_mask,
            caption="OcuNeuro AI Vessel Mask",
            width=500
        )


    st.success(
        "วิเคราะห์เส้นเลือดจอประสาทตาสำเร็จ"
    )


    # -----------------------------
    # Retinal Measurements
    # -----------------------------

    st.subheader(
        "🩸 Retinal Vessel Measurements"
    )

    metric1, metric2 = st.columns(2)

    with metric1:

        st.metric(
            "Vessel Density",
            f"{retinal_density * 100:.2f}%"
        )

    with metric2:

        st.metric(
            "Vessel Area",
            f"{int(retinal_area):,} pixels"
        )


# ==========================================
# Clinical AI
# ==========================================

st.header("2. Clinical Information")

age = st.number_input(
    "อายุ",
    min_value=1,
    max_value=120,
    value=60
)

hypertension = st.selectbox(
    "มีประวัติความดันโลหิตสูงหรือไม่?",
    ["ไม่มี", "มี"]
)

heart_disease = st.selectbox(
    "มีประวัติโรคหัวใจหรือไม่?",
    ["ไม่มี", "มี"]
)

avg_glucose_level = st.number_input(
    "ระดับน้ำตาลในเลือดเฉลี่ย",
    min_value=50.0,
    max_value=400.0,
    value=150.0
)

bmi = st.number_input(
    "BMI",
    min_value=10.0,
    max_value=60.0,
    value=28.1
)


if st.button("🧠 Analyze Risk"):

    user_data = {

        "age": age,

        "hypertension":
            1 if hypertension == "มี" else 0,

        "heart_disease":
            1 if heart_disease == "มี" else 0,

        "avg_glucose_level":
            avg_glucose_level,

        "bmi":
            bmi,

        "gender_Male": 1,
        "gender_Other": 0,

        "ever_married_Yes": 1,

        "work_type_Never_worked": 0,
        "work_type_Private": 1,
        "work_type_Self-employed": 0,
        "work_type_children": 0,

        "Residence_type_Urban": 1,

        "smoking_status_formerly smoked": 0,
        "smoking_status_never smoked": 1,
        "smoking_status_smokes": 0
    }


    user_df = pd.DataFrame(
        [user_data]
    )

    user_df = user_df[
        clinical_features
    ]


    probability = clinical_model.predict_proba(
        user_df
    )[0, 1]


    risk_percent = probability * 100


    # ==========================================
    # Result
    # ==========================================

    st.header("3. OcuNeuro OS Result")

    st.metric(
        "Clinical Risk Score",
        f"{risk_percent:.2f}%"
    )


    if probability >= clinical_threshold:

        st.warning(
            "Higher screening risk"
        )

    else:

        st.success(
            "Lower screening risk"
        )


    st.caption(
        "ผลลัพธ์นี้มาจาก Clinical AI Prototype "
        "และไม่ใช่การวินิจฉัยทางการแพทย์"
    )


    if retinal_density is not None:

        st.subheader(
            "Retinal AI Features"
        )

        st.write(
            f"Vessel Density: "
            f"{retinal_density * 100:.2f}%"
        )

        st.write(
            f"Vessel Area: "
            f"{int(retinal_area):,} pixels"
        )

        st.info(
            "Retinal features และ Clinical Risk "
            "แสดงแยกกันใน prototype นี้ "
            "ยังไม่ได้รวมเป็น multimodal stroke model"
        )
