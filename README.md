# PhysVidBench Inference Pipeline

This repository enables reproducibility of our results on the PhysVidBench benchmark. It involves generating videos from prompts, captioning them, and evaluating physical commonsense understanding using a question-answering pipeline.

---

## 🔁 Reproducing Results

To reproduce our results, please follow the steps below:

### 1. Video Generation

First, use your chosen text-to-video (T2V) model to generate videos based on each prompt in the `prompts_questions.csv` file.

### 2. Caption Extraction

For each generated video, extract **8 captions** describing different aspects of the video. We used the [**AuroraCap**](https://github.com/rese1f/aurora) model for this step.

To do this:
- Visit the AuroraCap model's GitHub repository.
- Replace its `inference.py` file with the `inference.py` file provided in this repository.
- This modified script will automatically use the captinonig prompts and produce 8 captions per video.

Save all captions in the `captions/` directory, following the same naming convention as used by the scripts.

### 3. Question-Answering over Captions

Once all captions are ready, run the question-answering module using Gemini:

```bash
python multi_ask.py
