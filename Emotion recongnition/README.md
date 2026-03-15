# Emotion Recognition App

This is a simple emotion recognition system built with **Streamlit** and the **`fer`** library. It lets you upload an image of a face and returns the **dominant emotion** with a confidence score and an annotated preview.

## 1. Setup

From the project folder:

```bash
pip install -r requirements.txt
```

If you are using `python` instead of `pip` directly:

```bash
python -m pip install -r requirements.txt
```

## 2. Run the app

From the same folder:

```bash
streamlit run app.py
```

After a few seconds, a browser window will open (or you can open the URL shown in the terminal, usually `http://localhost:8501`).

## 3. Using the app

- Upload a **JPG** or **PNG** image with at least one clearly visible face.
- Wait for the analysis to complete.
- You will see:
  - The **detected dominant emotion** (e.g. happy, sad, angry).
  - A **confidence score** between 0 and 1.
  - An **annotated image** with a bounding box and label.

## 4. Notes

- For best results, use images where the face is frontal and well-lit.
- The model is pre‑trained and may not be perfect; treat outputs as approximate.

