import os
import torch
import torchvision
import numpy as np
from PIL import Image
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, CLIPImageProcessor
from huggingface_hub import snapshot_download
from src.xtuner.xtuner.utils import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, PROMPT_TEMPLATE
from src.xtuner.xtuner.model.aurora import AuroraEncoder, AuroraModel



def process_text(inputs, tokenizer):
    ids = []
    for idx, chunk in enumerate(inputs.split(DEFAULT_IMAGE_TOKEN)):
        enc = tokenizer.encode(chunk, add_special_tokens=(idx == 0))
        ids.extend(enc)
        if idx != len(inputs.split(DEFAULT_IMAGE_TOKEN)) - 1:
            ids.append(IMAGE_TOKEN_INDEX)
    return torch.tensor(ids).cuda().unsqueeze(0)


def sample_frames(vframes, num_frames=8):
    idx = np.linspace(num_frames // 2, len(vframes) - num_frames // 2,
                      num_frames, dtype=int)
    return [torchvision.transforms.functional.to_pil_image(vframes[i]) for i in idx]


def predict_aurora(video_path, prompt_type):
    vframes, _, _ = torchvision.io.read_video(video_path, pts_unit="sec",
                                              output_format="TCHW")
    images = sample_frames(vframes, 8)

    image_tensor = image_processor(images, return_tensors='pt')['pixel_values']
    image_tensor = torch.stack([t.to(torch.float16).cuda() for t in image_tensor]).unsqueeze(0)

    prompt = " ".join([DEFAULT_IMAGE_TOKEN] * len(images)) + "\n" + prompt_type
    prompt_text = PROMPT_TEMPLATE.vicuna["INSTRUCTION"].format(input=prompt, round=1)
    input_ids = process_text(prompt_text, tokenizer)

    data = {"pixel_values": image_tensor, "input_ids": input_ids}
    auroracap.visual_encoder.reset_tome_r(0.8)

    outs = auroracap(data, mode="inference")
    cont = auroracap.llm.generate(**outs, do_sample=False, temperature=0.0,
                                  top_p=1.0, num_beams=1, max_new_tokens=2048)
    return tokenizer.batch_decode(cont, skip_special_tokens=True)[0]


BASE_PATH = "home-esanli25/benchmark_videos/"
MODEL_NAME = "aurora" 
BASE_OUT_DIR = f"{BASE_PATH}/outputs/{MODEL_NAME}"
os.makedirs(BASE_OUT_DIR, exist_ok=True)

pretrained_pth = snapshot_download(repo_id="wchai/AuroraCap-7B-VID-xtuner")
auroracap = AuroraModel(
    llm=AutoModelForCausalLM.from_pretrained(pretrained_pth, trust_remote_code=True,
                                             torch_dtype=torch.float16),
    visual_encoder=AuroraEncoder.from_pretrained(
        os.path.join(pretrained_pth, "visual_encoder"), torch_dtype=torch.float16
    )
).cuda()
auroracap.projector = AutoModel.from_pretrained(
    os.path.join(pretrained_pth, "projector"),
    torch_dtype=torch.float16, trust_remote_code=True
).cuda()

image_processor = CLIPImageProcessor.from_pretrained(
    "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
    trust_remote_code=True, size=378, crop_size=378
)
tokenizer = AutoTokenizer.from_pretrained(pretrained_pth, trust_remote_code=True,
                                          padding_side='right')



base_prompts = [
    "Describe the physical principles at work in the video, such as energy transfer, causal relationships, balance or imbalance, and any visible changes in physical state and etc. (e.g., solid to liquid).",
    "Describe the main objects in the video focusing on their material properties (e.g., rigid, soft, metallic), and what actions they allow or prevent and etc. (e.g., can be squeezed, can contain something).",
    "Describe the spatial layout and relationships in the video: object positions, orientations, fit between shapes, occlusions, and how geometry affects interactions and etc..",
    "Describe the sequence and timing of events in the video, including any delays, waiting periods, or causal orderings between actions or state changes and etc..",
    "Describe the actions performed in the video, focusing on the goals of the agent, the order of steps taken, and whether the actions appear intentional or methodical and etc..",
    "Describe how materials change or interact in the video: melting, freezing, breaking, mixing, or undergoing chemical or physical transformations and etc..",
    "Describe how forces are applied in the video (e.g., pushing, pulling, lifting), how objects respond (e.g., acceleration, resistance), and any indications of inertia or physical resistance and etc..",
    "Describe the video in detail"
]
instruction_suffix = (
    "Only describe what can be directly observed in the video. "
    "Do not make assumptions or include external knowledge that is not visually confirmed."
)
prompt_types = [p + instruction_suffix for p in base_prompts]
suffixes = ["_FP", "_OP", "_SR", "_TD", "_AU", "_MT", "_FM", ""]



variants_main = [
    "cosmos_14b_short"
]


for i in range(1):
    idx_str = f"{i}"
    for variant in variants_main:
        variant_out_dir = os.path.join(BASE_OUT_DIR, variant)
        os.makedirs(variant_out_dir, exist_ok=True)

        for s_idx, sfx in enumerate(suffixes):
            video_path = f"{BASE_PATH}/videos/{variant}/{i}.mp4"
            out_file   = os.path.join(variant_out_dir, f"{idx_str}{sfx}.txt")
            try:
                caption = predict_aurora(video_path, prompt_types[s_idx])
                with open(out_file, "w", encoding="utf-8") as f:
                    f.write(caption + "\n")
            except Exception as e:
                print(f"Error processing {video_path}: {e}")


