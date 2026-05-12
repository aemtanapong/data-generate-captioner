import streamlit as st
import pandas as pd
import numpy as np
import base64
import data_moving as data_radar
import data_intensity_v3 as data_intensity_v3
import matplotlib.pyplot as plt
import cv2
import imageio
import io
import geopandas as gpd
import time
import model_main
import plotly.express as px
import random
from datetime import datetime
import re
import data_coverage
plt.rcParams['font.family'] = 'Tahoma'

st.title("Radar Animation")
st.set_page_config( page_title="Radar Detail")

with st.status("generate caption", expanded=True) as n:
    uploaded_file = st.file_uploader("Upload Radar GIF", type=["gif", "webp"])
    
def ui_update(text, p):
    status.write(text)
    progress.progress(p)
def show_gif(gif_bytes, width=500):
    b64 = base64.b64encode(gif_bytes).decode()
    st.markdown(
        f'<div style="text-align:center;">'
        f'<img src="data:image/gif;base64,{b64}" width="{width}">'
        f'</div>',
        unsafe_allow_html=True
    )

    
if uploaded_file is not None:
    gif_bytes = uploaded_file.read()

    st.success("Radar uploaded ✅")
    progress = st.progress(0)
    # ui_update("กำลัง Upload ไฟล์",0)
    with st.status("🚀 Radar Animation", expanded=True) as status:
        print("data")
        # 👉 Replace with real computed values later
        direction = "NE"
        with st.spinner("กำลังประมวลผล..."):
            move_caption_data, render_data, output_frame = data_radar.radar_pipeline(
                uploaded_file,
                update=ui_update
            )
            print(direction)
            # -------------------------
            # Row 1: Information
            # -------------------------
            st.subheader("📊 Radar Information")
            col1, col2 = st.columns(2)

            with col1:
                st.metric("🧭 Direction", move_caption_data[1])

            with col2:
                try:
                    st.metric("📐 Degree", f"{move_caption_data[0]:.0f}°")
                except:
                    st.metric("📐 Degree", f"{move_caption_data[0]}°")

            st.divider()

            # -------------------------
            # Row 2: Radar A
            # -------------------------
            st.subheader("Radar (Original)")
            show_gif(gif_bytes)
            

            # -------------------------
            # Row 3: Radar B
            # -------------------------
            st.subheader("Radar (Prediction)")
            # แปลงเป็น RGB ถ้ามาจาก OpenCV
            render_rgb = cv2.cvtColor(render_data, cv2.COLOR_BGR2RGB)

            # แปลงเป็น base64
            _, buffer = cv2.imencode(".png", render_rgb)
            b64 = base64.b64encode(buffer).decode()

            st.markdown(
                f"""
                <div style="text-align:center;">
                    <img src="data:image/png;base64,{b64}" width="500">
                </div>
                """,
                unsafe_allow_html=True
            )

            # -------------------------
            # Row 4: Radar C
            # -------------------------
            st.subheader("Radar Detection Cluster")

            gif_buffer = io.BytesIO()

            imageio.mimsave(
                gif_buffer,
                output_frame,
                format="GIF",
                fps=10,
                loop=0
            )

            gif_buffer.seek(0)

            st.image(gif_buffer.read(), caption="Radar Animation")
            # show_gif(gif_bytes)
            # data_intensity_v3.get_district_name_rain(gif_buffer)
            print("data")
            st.success("Done!")
    with st.status("Coverage คำนวนคลอบคลุมฝน", expanded=True) as status:
        rgb, mask_vis ,cropped, rain_vis, highlight, plot_img, average_coverage = data_coverage.get_data(uploaded_file)
        print("data")
        st.subheader("📊 Radar Information")
        st.metric("🧭 Coverage", f"{average_coverage:.2f}%")
        descibe_coverage_mask, rain_data_debug = st.columns(2)
        
        with descibe_coverage_mask:
            st.image(rgb, caption="RGB",  use_container_width=True)
            st.image(mask_vis, caption="Mask", use_container_width=True)
            st.image(cropped, caption="Cropped", use_container_width=True)
        with rain_data_debug:
            st.image(rain_vis, caption="Rain Visualization", use_container_width=True)
            st.image(highlight, caption="Highlight", use_container_width=True)
            st.image(plot_img, caption="plot_img", use_container_width=True)
        
        st.success(f"Done!")
