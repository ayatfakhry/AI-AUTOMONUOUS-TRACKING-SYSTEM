# 🛰️ OrbitalMind — Ground Station Mission Control

Real-time satellite tracking dashboard for the OrbitalMind autonomous ground station.
Beni Suef University · ECE / Satellite Navigation & Space Technology.

Live TLEs (CelesTrak) → **SGP4 propagation (Skyfield)** → look angles (Az/El) →
ESP32 motor command. Cross-validated against STK (Az error ~0.021°).

## Pages
- **Command Center** — live sky view + ground track + key metrics
- **Vision Pipeline** — photo of printed satellite image → detect → SGP4 → predict +60s → ESP32 command
- **Live Tracking** — Az/El look angles + "Send GOTO to ESP32" button
- **Sky Map** — observer-centric polar plot of all catalog objects
- **Satellite Catalog** — full tracked object list with live status
- **Pass Predictions** — upcoming passes over Beni Suef (next 24 h)
- **Ground Station** — hardware telemetry (motors, I2C, compass, PID, calibration)

---

## 🟢 الطريقة (خطوة بخطوة)

### الملفات
انتي محتاجة الملفين دول بس في نفس الفولدر/الـ repo:
- `app.py`
- `requirements.txt`

**مفيش أي تعديل لازم تعمليه** عشان يشتغل. بس لو حبيتي تغيّري الأقمار الاحتياطية
عدّلي `FALLBACK_TLES` فوق في `app.py`. محطة بني سويف مظبوطة خلاص.

### 1) رفع على GitHub (زي داشبورد الـ NYC بالظبط)
- روحي على الـ repo بتاعك (مثلاً `ayatfakhry`) → **Add file → Upload files**
- ارفعي `app.py` و `requirements.txt` زي ما هما → **Commit**

### 2) Deploy على Streamlit Cloud
- **share.streamlit.io** → **New app**
- اختاري: الـ repo · الـ branch (`main`) · Main file path = `app.py`
- **Deploy** → هيديكي رابط `https://<name>.streamlit.app`

### 3) تشغيل محلي (Local) — لازم للـ ESP32 الحقيقي
```bash
pip install -r requirements.txt
streamlit run app.py
```
هيفتح على `http://localhost:8501`

---

## ⚠️ نقطة مهمة: الفرق بين الأونلاين والـ local

| | Streamlit Cloud (أونلاين) | Local (لابتوبك) |
|---|---|---|
| Tracking / SGP4 / Vision / Passes | ✅ | ✅ |
| توصيل ESP32 الحقيقي على COM4 | ❌ (السيرفر مبيوصلش USB) | ✅ |

يعني: الرابط الأونلاين ممتاز للـ **defense والعرض**، وتشغيلك local على اللابتوب
هو اللي بيحرّك الموتور فعلاً.

### توصيل الـ ESP32
لما تشغّلي local: من الـ sidebar → **🔌 ESP32 Link** → اكتبي `COM4` → **Connect**.
بعدها زراير "Send GOTO to ESP32" في صفحتي Live Tracking و Vision Pipeline
هتبعت أمر `GOTO AZ=.. EL=..` على السيريال. الفيرموير عندك لازم يفهم الأمر ده.
