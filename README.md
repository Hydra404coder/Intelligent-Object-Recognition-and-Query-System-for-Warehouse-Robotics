# 🤖 Intelligent Warehouse Robotics Assistant
### Combining Computer Vision & Retrieval-Augmented Generation for Smart Item Handling

> 📋 **Built as a solution to the AI Research Intern — Technical Assessment Task**
> This project demonstrates an end-to-end AI system that classifies warehouse items using a fine-tuned CNN and retrieves category-specific handling instructions from operational manuals using a RAG pipeline.

---

## 🎯 Problem Statement

Design an intelligent warehouse robotics system that can:
1. **Visually identify** the category of an item (fragile, hazardous, or heavy) from an image
2. **Retrieve context-aware handling instructions** from operational manuals using natural language queries
3. **Integrate both systems** into a unified pipeline — classify first, then instruct

---

## 🏗️ System Architecture

<img width="857" height="438" alt="image" src="https://github.com/user-attachments/assets/0f8a5a53-9f8a-4f4a-b23f-a1a7d30e6afe" />


---

## 📊 Model Performance

### Classification Report (Validation Set)

| Category    | Precision | Recall | F1-Score | Support |
|-------------|-----------|--------|----------|---------|
| 🔻 Fragile   | 0.86      | 1.00   | 0.92     | 6       |
| ☣️ Hazardous | 1.00      | 0.80   | 0.89     | 5       |
| 🏋️ Heavy     | 1.00      | 1.00   | 1.00     | 7       |
| **Accuracy** |           |        | **0.94** | **18**  |
| Macro Avg   | 0.95      | 0.93   | 0.94     | 18      |
| Weighted Avg| 0.95      | 0.94   | 0.94     | 18      |

### Training Progress (15 Epochs)

<img width="1026" height="599" alt="image" src="https://github.com/user-attachments/assets/dc0f43ab-b1ee-410e-b6e7-89f71707bb18" />


---

## 🔧 Technical Stack

| Layer            | Technology                          | Purpose                          |
|------------------|-------------------------------------|----------------------------------|
| **Deep Learning** | PyTorch, TorchVision               | CNN training & inference         |
| **CV Model**      | MobileNetV2 (fine-tuned)           | Image classification             |
| **Embeddings**    | Sentence Transformers (MiniLM-L6)  | Semantic text representation     |
| **Vector DB**     | FAISS (IndexFlatL2)                | Similarity search                |
| **LLM**           | Google Gemini 2.5-Flash            | Response generation              |
| **Doc Parsing**   | PyPDF2                             | PDF text extraction              |
| **Visualization** | OpenCV, Matplotlib, Seaborn        | Bounding boxes & metrics plots   |
| **UI**            | ipywidgets                         | Interactive notebook interface    |

---


---

## 🖼️ Visual Output

The system provides visual feedback with:
- ✅ **Bounding box** drawn around the classified item (OpenCV)
- ✅ **Category label** overlaid on the image
- ✅ **Confusion matrix heatmap** for model evaluation
- ✅ **Interactive widgets** for image upload & question input

---

## 🚀 Quick Start

```bash
# Install dependencies
pip install torch torchvision scikit-learn matplotlib seaborn
pip install google-genai faiss-cpu PyPDF2 sentence-transformers opencv-python

# Run the notebook
jupyter notebook [Vs_code_llm.ipynb](http://_vscodecontentref_/1)
Run all cells sequentially to train the model
Upload an image using the interactive widget
Ask a question about the detected item category
Receive AI-generated handling instructions from the manuals





