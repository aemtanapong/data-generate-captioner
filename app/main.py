import streamlit as st
import pandas as pd
import numpy as np
import base64
import app.data_moving as data_radar
import app.data_intensity_v3 as data_intensity_v3
import matplotlib.pyplot as plt
import cv2
import imageio
import io
import geopandas as gpd
import time
plt.rcParams['font.family'] = 'Tahoma'
# with st.sidebar:
#     with st.echo():
#         st.write("This code will be printed to the sidebar.")

st.title("Radar Animation")
st.set_page_config( page_title="Radar Detail")
import plotly.express as px
uploaded_file = st.file_uploader("Upload Radar GIF", type=["gif", "webp"])
# def ui_update(text, p):
#     status.write(text)
#     progress.progress(p)
# def show_gif(gif_bytes, width=500):
#     b64 = base64.b64encode(gif_bytes).decode()
#     st.markdown(
#         f'<div style="text-align:center;">'
#         f'<img src="data:image/gif;base64,{b64}" width="{width}">'
#         f'</div>',
#         unsafe_allow_html=True
#     )

    
# if uploaded_file is not None:
#     gif_bytes = uploaded_file.read()

#     st.success("Radar uploaded ✅")
#     progress = st.progress(0)
#     # ui_update("กำลัง Upload ไฟล์",0)
#     with st.status("🚀 Radar Animation", expanded=True) as status:
#         print("data")
#         # 👉 Replace with real computed values later
#         direction = "NE"
#         with st.spinner("กำลังประมวลผล..."):
#             move_caption_data, render_data, output_frame = data_radar.radar_pipeline(
#                 uploaded_file,
#                 update=ui_update
#             )
#             print(direction)
#             # -------------------------
#             # Row 1: Information
#             # -------------------------
#             st.subheader("📊 Radar Information")
#             col1, col2 = st.columns(2)

#             with col1:
#                 st.metric("🧭 Direction", move_caption_data[1])

#             with col2:
#                 try:
#                     st.metric("📐 Degree", f"{move_caption_data[0]:.0f}°")
#                 except:
#                     st.metric("📐 Degree", f"{move_caption_data[0]}°")

#             st.divider()

#             # -------------------------
#             # Row 2: Radar A
#             # -------------------------
#             st.subheader("Radar (Original)")
#             show_gif(gif_bytes)
            

#             # -------------------------
#             # Row 3: Radar B
#             # -------------------------
#             st.subheader("Radar (Prediction)")
#             # แปลงเป็น RGB ถ้ามาจาก OpenCV
#             render_rgb = cv2.cvtColor(render_data, cv2.COLOR_BGR2RGB)

#             # แปลงเป็น base64
#             _, buffer = cv2.imencode(".png", render_rgb)
#             b64 = base64.b64encode(buffer).decode()

#             st.markdown(
#                 f"""
#                 <div style="text-align:center;">
#                     <img src="data:image/png;base64,{b64}" width="500">
#                 </div>
#                 """,
#                 unsafe_allow_html=True
#             )

#             # -------------------------
#             # Row 4: Radar C
#             # -------------------------
#             st.subheader("Radar Detection Cluster")

#             gif_buffer = io.BytesIO()

#             imageio.mimsave(
#                 gif_buffer,
#                 output_frame,
#                 format="GIF",
#                 fps=10,
#                 loop=0
#             )

#             gif_buffer.seek(0)

#             st.image(gif_buffer.read(), caption="Radar Animation")
#             # show_gif(gif_bytes)
#             # data_intensity_v3.get_district_name_rain(gif_buffer)
#             print("data")
#             st.success("Done!")
        

#     with st.status("🌧️ Interactive Radar Data", expanded=True) as status:
#         print("data")
#         with st.spinner("กำลังประมวลผล..."):
#             st.title("🌧️ Interactive Radar Data")
#             progress = st.progress(0.3)
#             status_text = st.empty()
            
            

#             df_rain_intensity = data_intensity_v3.generate_data(uploaded_file, ui_update)
#             df = df_rain_intensity['data_rain_level_frame']
#             # -------------------------
#             # PROGRESS BAR
#             # -------------------------
            
#             # -------------------------
#             # LAYOUT
#             # -------------------------
#             col1, col2, col3 = st.columns(3)

#             # ==================================================
#             # 📈 LINE
#             # ==================================================
#             # with col1:
#             st.subheader("📈 Trend")
#             st.caption("ดูการเพิ่ม-ลดของฝนในแต่ละเขต")

#             fig1 = px.line(
#                 df,
#                 x="Frame",
#                 y="Score",
#                 color="District",
#                 markers=True
#             )

#             st.plotly_chart(fig1, use_container_width=True)



#             # ==================================================
#             # 📊 BAR
#             # ==================================================
#             # with col2:
#             st.subheader("🏆 Avg Score")

#             avg = df.groupby("District", as_index=False)["Score"].mean()

#             fig2 = px.bar(
#                 avg,
#                 x="District",
#                 y="Score",
#                 color="Score"
#             )

#             st.plotly_chart(fig2, use_container_width=True)



#             # ==================================================
#             # 🔥 HEATMAP
#             # ==================================================
#             # with col3:
#             st.subheader("🔥 Heatmap")

#             pivot = df.pivot_table(
#                 index="District",
#                 columns="Frame",
#                 values="Score",
#                 fill_value=0
#             )

#             fig3 = px.imshow(
#                 pivot,
#                 text_auto=True,
#                 color_continuous_scale="YlOrRd"
#             )

#             st.plotly_chart(fig3, use_container_width=True)


#             ui_update("✅สำเร็จ", 1.0)
#             st.success("🗺️ Mock Radar District Map")
#     with st.status("🌧️ Interactive Radar Data", expanded=True) as status:
#          with st.spinner("กำลังประมวลผล..."):
#             st.title("🗺️ Mock Radar District Map")
#             progress = st.progress(0.3)
#             status_text = st.empty()
#             print(df_rain_intensity)

#                         # -------------------------
#             # LOAD DATA
#             # -------------------------
#             gdf = gpd.read_file("mapdata/Export_Output.shp")

#             df = df_rain_intensity['data_rain_level_frame']
#             summary = df.groupby("District", as_index=False)["Score"].mean()

#             # rename ให้ตรง
#             gdf = gdf.rename(columns={"DISTRICT_T": "District"})

#             # merge
#             map_df = gdf.merge(summary, on="District", how="left")
#             map_df["Score"] = map_df["Score"].fillna(0)

#             # -------------------------
#             # LEVEL
#             # -------------------------
#             map_df["level"] = pd.cut(
#                 map_df["Score"],
#                 bins=[0, 15, 30, 100],
#                 labels=["Light", "Medium", "Heavy"]
#             )

#             colors = {
#                 "Light": "green",
#                 "Medium": "orange",
#                 "Heavy": "red"
#             }

#             # -------------------------
#             # PLOT (ครั้งเดียวพอ)
#             # -------------------------
#             fig, ax = plt.subplots(figsize=(10, 10))

#             # เขตไม่มีฝน (background)
#             map_df[map_df["Score"] == 0].plot(
#                 ax=ax,
#                 color="lightgrey",
#                 edgecolor="black"
#             )

#             # เขตมีฝน (แบ่ง level)
#             for lvl, color in colors.items():
#                 subset = map_df[map_df["level"] == lvl]
#                 subset.plot(
#                     ax=ax,
#                     color=color,
#                     label=lvl,
#                     edgecolor="black"
#                 )

#             # -------------------------
#             # LABEL (เฉพาะมีฝน)
#             # -------------------------
#             rain = map_df[map_df["Score"] > 0]

#             for _, row in rain.iterrows():
#                 if row.geometry is not None:
#                     x = row.geometry.centroid.x
#                     y = row.geometry.centroid.y
#                     ax.text(x, y, row["District"], fontsize=7, ha="center")

#             # -------------------------
#             # STYLE
#             # -------------------------
#             ax.set_title("🌧️ Rain Map by District", fontsize=14)
#             ax.axis("off")
#             ax.legend()

#             # -------------------------
#             # STREAMLIT
#             # -------------------------
#             st.pyplot(fig)

#             rain_df = summary[summary["Score"] > 0].sort_values("Score", ascending=False)

#             top = rain_df.head(10)

#             fig3, ax3 = plt.subplots(figsize=(8, 5))
#             ax3.barh(top["District"], top["Score"])

#             ax3.set_title("Top 10 Rain Districts")
#             ax3.invert_yaxis()

#             st.pyplot(fig3)

#             rain_only = df[df["Score"] > 0]

#             fig4, ax4 = plt.subplots(figsize=(10, 5))

#             for d in rain_only["District"].unique():
#                 sub = rain_only[rain_only["District"] == d]
#                 ax4.plot(sub["Frame"], sub["Score"], label=d)

#             ax4.set_title("Rain Trend by District Name")
#             ax4.set_xlabel("Frame")
#             ax4.set_ylabel("Score")

#             # ถ้าเขตเยอะ → ไม่ต้อง show legend
#             # ax4.legend()

#             st.pyplot(fig4)
    # st.subheader("📊 Raw Data")
    # st.dataframe(rain_intensity["raw"])

    # st.subheader("🌧️ Rain Areas")
    # st.dataframe(result["raining"])

    # st.subheader("📈 Trend")
    # st.dataframe(result["trend"])

    # st.subheader("🌡️ Intensity")
    # st.dataframe(result["intensity"])
    