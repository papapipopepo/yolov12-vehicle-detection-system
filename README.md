🚗 Vehicle Detection  YOLOv12
---

> Smart Traffic Vehicle Detection & Density Analysis\
> Powered by YOLOv12 + Streamlit + Supervision

* * * * *

📌 Overview
-----------

This project implements a **vehicle detection and traffic density analysis system** using **YOLOv12**.\
The application is built with **Streamlit** and supports:

-   🚗 Vehicle detection (Bus, Car, Van)

-   📊 Automatic vehicle counting

-   📈 Traffic density classification (LOW / MEDIUM / HIGH)

-   🖼️ Single image detection

-   🎬 Video detection

-   📦 Batch image processing

-   📜 Detection history tracking

-   📊 Interactive visual analytics (Plotly)

* * * * *

🧠 Model Information
--------------------

-   Architecture: **YOLOv12**

-   Classes:

    -   Bus

    -   Car

    -   Van

-   Input size (inference): `1024`

-   Evaluation Performance:

    -   mAP@50: **~0.88**

    -   mAP@50--95: **~0.69**

* * * * *

🚦 Traffic Density Logic
------------------------

Traffic density is calculated based on total detected vehicles:

| Total Vehicles | Density |
| --- | --- |
| ≤ 3 | LOW 🟢 |
| 4 -- 7 | MEDIUM 🟡 |
| > 7 | HIGH 🔴 |

* * * * *

🏗️ Project Structure
---------------------
```
PURWA_YOLO/
│
├── models/
│   └── best_vehicle.pt
│
├── src/
│   └── purwa_yolo/
│       ├── __init__.py
│       └── main.py
│
├── training_code/
│   └── Train_YOLOv12_Vehicle.ipynb
│
├── tests/
│
├── dockerfile
├── docker-compose.yml
├── pyproject.toml
├── poetry.lock
└── README.md
```
* * * * *

⚙️ Installation
---------------

### 1️⃣ Clone Repository
```
git clone https://github.com/papapipopepo/yolov12-vehicle-detection-system.git
cd yolov12-vehicle-detection-system
```
### 2️⃣ Install Dependencies (Poetry)
```
poetry install
```
Or using pip:
```
pip install streamlit ultralytics supervision opencv-python pillow plotly pandas
```
* * * * *

▶️ Run the Application
----------------------

If using Poetry:
```
poetry run streamlit run src/purwa_yolo/main.py
```
Without Poetry:
```
streamlit run src/purwa_yolo/main.py
```
App will run at:
```
http://localhost:8501
```
* * * * *

🖼️ Features
------------

### 1️⃣ Single Image Detection

-   Upload image

-   Adjust confidence & IoU

-   Download annotated result

-   Export CSV summary

-   Visual bar chart distribution

* * * * *

### 2️⃣ Video Detection

-   Frame-by-frame processing

-   Adjustable frame skipping

-   Real-time metrics

-   Line chart for vehicle trends

* * * * *

### 3️⃣ Batch Processing

-   Upload multiple images

-   Automatic summary table

-   Batch analytics chart

-   CSV export

* * * * *

### 4️⃣ Session History

-   Stores detection history

-   Trend visualization

-   Export history CSV

* * * * *

🎯 Technical Highlights
-----------------------

-   Uses `sv.Detections.from_ultralytics()` for conversion

-   Applies Non-Maximum Suppression (NMS)

-   Uses caching with `@st.cache_resource`

-   Custom dark UI with advanced CSS styling

-   Interactive visualization using Plotly

* * * * *

📊 Example Output
-----------------

-   Bounding boxes with confidence score

-   Vehicle count per class

-   Total vehicle count

-   Density badge (LOW / MEDIUM / HIGH)

-   Distribution bar chart

-   Detection trend over time

* * * * *

🚀 Future Improvements
----------------------

-   🚦 Traffic jam length estimation

-   🛣️ Lane-based vehicle counting

-   📍 Object tracking (SORT / ByteTrack)

-   📡 Real-time CCTV streaming

-   🧠 Small object optimization with tiling inference

* * * * *

👤 Author
---------

**Ezra Satria Bagas Airlangga**  
Master’s Student – Electrical Engineering, Telkom University  
📧 ezra.satria16@gmail.com
🔗[LinkedIn](https://linkedin.com/in/ezrasatriabagas/)

#ComputerVision #DeepLearning #YOLOv12
#VehicleDetection #TrafficAnalysis
#StreamlitApp #PythonProject
#CapstoneProject #Purwadhika
