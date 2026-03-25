# import torch
# from transformers import T5ForConditionalGeneration, T5Tokenizer
# from peft import LoraConfig, get_peft_model, TaskType

# device = "mps" if torch.backends.mps.is_available() else "cpu"

# MODEL_PATH = "../outputs/model"   # your supervised trained model

# print("Loading base model...")
# model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH).to(device)

# tokenizer = T5Tokenizer.from_pretrained("t5-small")

# # ---------------- LoRA CONFIG ----------------
# lora_config = LoraConfig(
#     r=8,                       # rank (small brain attachment)
#     lora_alpha=16,
#     target_modules=["q", "v"], # attention matrices only
#     lora_dropout=0.05,
#     bias="none",
#     task_type=TaskType.SEQ_2_SEQ_LM
# )

# print("Attaching LoRA adapters...")
# model = get_peft_model(model, lora_config)

# model.print_trainable_parameters()

# print("READY ✔ LoRA model loaded")

# ****************** task 5 @#$%^&*I(O)(*&^%$#$%^&*(*&^%$#$%^&*^%$#%^)
# )
# 
# 
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from peft import LoraConfig, get_peft_model, TaskType

# ---------------- DEVICE SETUP ----------------
device = "mps" if torch.backends.mps.is_available() else "cpu"

MODEL_PATH = "../outputs/model"

# ---------------- LOAD TOKENIZER ----------------
tokenizer = T5Tokenizer.from_pretrained("t5-small")

# ---------------- LOAD MODEL WITH QUANTIZATION ----------------
def load_model(quantization=None):
    print(f"Loading model with quantization = {quantization}")

    if quantization == "int8":
        model = T5ForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            load_in_8bit=True,
            device_map="auto"
        )

    elif quantization == "int4":
        model = T5ForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            load_in_4bit=True,
            device_map="auto"
        )

    else:  # fp32
        model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH).to(device)

    return model


# 👉 CHANGE THIS VALUE TO TEST
QUANTIZATION = "int8"   # options: None, "int8", "int4"

model = load_model(QUANTIZATION)


# ---------------- LoRA CONFIG ----------------
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q", "v"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM
)

print("Attaching LoRA adapters...")
model = get_peft_model(model, lora_config)

model.print_trainable_parameters()

print("READY ✔ LoRA + Quantized model loaded") 