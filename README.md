# Vision-G1: Towards General Vision Language Reasoning with Multi-Domain Data Curation
## Introduction
We present the reasoning VLM: Vision-G1, which is trained through multi-domain data curation. Specifically, we include training data from 46 data sources across 8 dimensions. This repo includes training code, data preprocessing scripts, and evaluation scripts.
## Installation
To enable the inference of Vision-G1, you can simply install the latest transformers library:
```bash
pip install transformers
```
Optionally, if you wish to accelerate the inference, please install vllm:
```bash
pip install vllm
```
To support training, installing verl is required:
```bash
cd training/
pip install -r requirements.txt
pip install flash_attn==2.7.4.post1 --no-build-isolation
pip install -e .
```
## Inference
Our model follows the transformers format. You can load it with standard transformers APIs:
```python
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "yzha/vision-g1",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)

# default processer
processor = AutoProcessor.from_pretrained("yzha/vision-g1")

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "<path to image>",
            },
            {"type": "text", "text": "<Question>"},
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)

```
For faster inference, the model also supports vllm server. Simply start a vllm host by:
```bash
vllm serve yzha/vision-g1 \
  --host 0.0.0.0 --port 8000 \
  --max-model-len 8192 \
```
## Training
The training scripts are in `training/examples/grpo_trainer`. For single node training, please use `single_node_vision_g1.sh`. For distributed training, please use `distributed_vision_g1.sh`. Before launching the training, you need to specify the training and validation data path in the bash script, by setting `train_files` and `test_files`. To launch the training, 
### Single node
```bash
cd training/
bash examples/grpo_trainer/single_node_vision_g1.sh
```
### Multiple nodes
```bash
cd training/
sbatch examples/grpo_trainer/distributed_vision_g1.sh
```
## Data
TBD
## Acknowledgement
We use [verl](https://github.com/volcengine/verl) as the codebase to build our training framework.
## Citation
If you are interested in our work, please cite:
```
@article{zha2025vision,
  title={Vision-G1: Towards General Vision Language Reasoning with Multi-Domain Data Curation},
  author={Zha, Yuheng and Zhou, Kun and Wu, Yujia and Wang, Yushu and Feng, Jie and Xu, Zhi and Hao, Shibo and Liu, Zhengzhong and Xing, Eric P and Hu, Zhiting},
  journal={arXiv preprint arXiv:2508.12680},
  year={2025}
}
```
