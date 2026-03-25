import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers.generation.logits_process import LogitsProcessorList
from trl import PPOTrainer, PPOConfig, AutoModelForSeq2SeqLMWithValueHead
from peft import PeftModel
import os, sys, random
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from execution_reward_soft import execution_reward_soft_batch_parallel_by_db
from execution_reward import extract_tables, extract_columns
from constrained_decoding import SchemaConstrainedLogitsProcessor
try:
    from src.prompting import get_schema_text
except Exception:  # pragma: no cover
    get_schema_text = None

# ======================================================
# DEVICE
# ======================================================
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ======================================================
# SETTINGS
# ======================================================
NUM_EPOCHS = 15
LOG_EVERY = 20
MAX_OUTPUT_TOKENS = 64
ROLLOUTS_PER_EPOCH = 1024
USE_CONSTRAINED_DECODING = os.environ.get("CONSTRAINED_DECODING", "0") == "1"
MAX_INPUT_TOKENS = 512
REWARD_MAX_WORKERS = int(os.environ.get("REWARD_MAX_WORKERS", "20"))
SOFT_SAMPLE_K = int(os.environ.get("SOFT_REWARD_SAMPLE_K", "10"))

# ======================================================
# PATHS
# ======================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_ROOT = os.path.join(PROJECT_ROOT, "data/database")

BASE_MODEL = "Salesforce/codet5-base"
ADAPTER_PATH = os.path.join(PROJECT_ROOT, "checkpoints/sft_adapter_codet5")

# ======================================================
# TOKENIZER
# ======================================================
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ======================================================
# MODEL
# ======================================================
model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(BASE_MODEL).to(device)
model.pretrained_model = PeftModel.from_pretrained(
    model.pretrained_model, ADAPTER_PATH, is_trainable=True
)

ref_model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(BASE_MODEL).to(device)
ref_model.pretrained_model = PeftModel.from_pretrained(
    ref_model.pretrained_model, ADAPTER_PATH, is_trainable=False
)
ref_model.eval()

# ======================================================
# PPO CONFIG
# ======================================================
PPO_LR = float(os.environ.get("PPO_LR", "2e-6"))
PPO_INIT_KL = float(os.environ.get("PPO_INIT_KL", "0.2"))
PPO_TARGET_KL = float(os.environ.get("PPO_TARGET_KL", "0.2"))

ppo_config = PPOConfig(
    learning_rate=PPO_LR,
    batch_size=8,
    mini_batch_size=2,
    ppo_epochs=2,
    init_kl_coef=PPO_INIT_KL,
    adap_kl_ctrl=True,
    kl_penalty="kl",
    target_kl=PPO_TARGET_KL,
    whiten_rewards=True,
    max_grad_norm=1.0,
)

trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    ref_model=ref_model,
    tokenizer=tokenizer,
)

# ======================================================
# DATA
# ======================================================
dataset = load_dataset("spider", split="train")

def sample_example():
    return dataset[random.randrange(len(dataset))]

def get_db_path(db_id):
    return os.path.join(DB_ROOT, db_id, f"{db_id}.sqlite")

# ======================================================
# GENERATION CONFIG
# ======================================================
generation_kwargs = dict(
    max_new_tokens=64,
    do_sample=True,
    temperature=0.3,
    top_p=0.7,
    top_k=20,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

# ======================================================
# TRAIN LOOP
# ======================================================
print("🚀 Starting SOFT RL training")

best_reward = -1e9

def encode_prompt_ids(question: str, db_id: str, schema_text: str) -> torch.Tensor:
    """
    Builds a prompt that always preserves the 'SQL:' suffix, truncating schema only
    to fit within MAX_INPUT_TOKENS.
    """
    prefix = (
        "You are a SQLite expert.\n\n"
        f"Database: {db_id}\n\n"
        "Schema:\n"
    )
    mid = "\n\nQuestion:\n"
    suffix = f"{question}\n\nSQL:"

    # Use the tokenizer call API with truncation to avoid warnings when strings
    # exceed tokenizer.model_max_length; we still do our own schema-only trimming below.
    prefix_ids = tokenizer(prefix, add_special_tokens=False, truncation=True, max_length=MAX_INPUT_TOKENS).input_ids
    schema_ids = tokenizer(schema_text or "", add_special_tokens=False, truncation=True, max_length=MAX_INPUT_TOKENS).input_ids
    mid_ids = tokenizer(mid, add_special_tokens=False, truncation=True, max_length=MAX_INPUT_TOKENS).input_ids
    suffix_ids = tokenizer(suffix, add_special_tokens=False, truncation=True, max_length=MAX_INPUT_TOKENS).input_ids

    eos_id = tokenizer.eos_token_id
    max_without_eos = MAX_INPUT_TOKENS - (1 if eos_id is not None else 0)

    fixed_len = len(prefix_ids) + len(mid_ids) + len(suffix_ids)
    if fixed_len > max_without_eos:
        # As a last resort, truncate the question/suffix portion (keeps "SQL:" at end).
        keep = max(0, max_without_eos - (len(prefix_ids) + len(mid_ids)))
        suffix_ids = suffix_ids[:keep]
        fixed_len = len(prefix_ids) + len(mid_ids) + len(suffix_ids)

    remaining_for_schema = max_without_eos - fixed_len
    if remaining_for_schema < 0:
        remaining_for_schema = 0
    schema_ids = schema_ids[:remaining_for_schema]

    ids = (prefix_ids + schema_ids + mid_ids + suffix_ids)[:max_without_eos]
    if eos_id is not None:
        ids = ids + [eos_id]

    return torch.tensor(ids, dtype=torch.long).to(device)

for epoch in range(1, NUM_EPOCHS + 1):

    epoch_reward_sum = 0
    valid_sql_count = 0
    total_seen = 0

    step_iterator = tqdm(
        range(0, ROLLOUTS_PER_EPOCH, ppo_config.batch_size),
        desc=f"Epoch {epoch}/{NUM_EPOCHS}"
    )

    for step in step_iterator:

        batch_meta = []
        query_tensors = []

        for _ in range(ppo_config.batch_size):
            ex = sample_example()
            q = ex["question"]
            gold = ex["query"]
            db_id = ex["db_id"]
            db_path = get_db_path(db_id)

            schema_text = ""
            if get_schema_text is not None:
                try:
                    schema_text = get_schema_text(db_id)
                except Exception:
                    schema_text = ""

            query_tensors.append(encode_prompt_ids(q, db_id, schema_text))
            batch_meta.append((q, gold, db_path, db_id))

        response_tensors = []

        with torch.no_grad():
            for i in range(len(query_tensors)):
                db_path = batch_meta[i][2]

                logits_processor = None
                if USE_CONSTRAINED_DECODING:
                    logits_processor = LogitsProcessorList([
                        SchemaConstrainedLogitsProcessor(tokenizer, db_path)
                    ])

                res = trainer.generate(
                    [query_tensors[i]],
                    logits_processor=logits_processor,
                    **generation_kwargs
                )

                response_tensors.append(res[0])

        batch_responses_text = []
        base_rewards = [None] * len(response_tensors)  # type: ignore[var-annotated]
        bonuses = [0.0 for _ in range(len(response_tensors))]
        apply_low_base_bonus = [False for _ in range(len(response_tensors))]

        needs_soft = []
        soft_rollouts = []

        for i in range(len(response_tensors)):
            response = tokenizer.decode(response_tensors[i], skip_special_tokens=True)
            batch_responses_text.append(response)

            question, gold_sql, db_path, db_id = batch_meta[i]
            total_seen += 1

            response_lower = response.lower()

            has_select = "select" in response_lower
            has_from = "from" in response_lower

            # ==================================================
            # BASE REWARD (defer execution-based part)
            # ==================================================
            if not has_select or not has_from:
                base_rewards[i] = -0.05
            elif "select " not in response_lower or " from " not in response_lower:
                base_rewards[i] = 0.0
            elif len(response.split()) < 5:
                base_rewards[i] = 0.0
            else:
                base_rewards[i] = None
                needs_soft.append(i)
                soft_rollouts.append((response, db_path, gold_sql))
                apply_low_base_bonus[i] = True
                if "where" in response_lower:
                    bonuses[i] += 0.12
                if "join" in response_lower:
                    bonuses[i] += 0.12

            # ==================================================
            # TOKEN-LEVEL REWARD
            # ==================================================
            gold_tokens = gold_sql.lower().split()
            pred_tokens = response_lower.split()
            common = 0
            for j in range(min(len(gold_tokens), len(pred_tokens))):
                if gold_tokens[j] == pred_tokens[j]:
                    common += 1
                else:
                    break
            if common >= 3 and gold_tokens:
                bonuses[i] += 0.03 * (common / len(gold_tokens))

            # ==================================================
            # TABLE/COLUMN BONUS
            # ==================================================
            pred_tables = set(extract_tables(response))
            gold_tables = set(extract_tables(gold_sql))
            if gold_tables:
                bonuses[i] += 0.25 * (len(pred_tables & gold_tables) / len(gold_tables))

            pred_cols = set(extract_columns(response))
            gold_cols = set(extract_columns(gold_sql))
            if gold_cols:
                bonuses[i] += 0.15 * (len(pred_cols & gold_cols) / len(gold_cols))

        # ==================================================
        # Task 1 integration: parallel soft execution reward (base)
        # ==================================================
        if soft_rollouts:
            soft_vals = execution_reward_soft_batch_parallel_by_db(
                soft_rollouts,
                max_workers=REWARD_MAX_WORKERS,
                sample_k=SOFT_SAMPLE_K,
            )
            for local_idx, base_r in zip(needs_soft, soft_vals):
                base_rewards[local_idx] = float(base_r)

        # ==================================================
        # Final reward composition
        # ==================================================
        batch_rewards = []
        batch_reward_sum = 0.0
        batch_valid_count = 0
        for i in range(len(response_tensors)):
            base = float(base_rewards[i] if base_rewards[i] is not None else 0.0)
            if apply_low_base_bonus[i] and base < 0.3:
                bonuses[i] += 0.15
            reward = max(-1.0, min(1.0, base + bonuses[i]))
            batch_rewards.append(torch.tensor(reward).to(device))
            batch_reward_sum += reward
            batch_valid_count += 1

        epoch_reward_sum += batch_reward_sum
        valid_sql_count += batch_valid_count

        trainer.step(query_tensors, response_tensors, batch_rewards)

        avg = epoch_reward_sum / max(valid_sql_count, 1)

        step_iterator.set_postfix({
            "Avg Reward": f"{avg:.3f}",
            "Valid": f"{valid_sql_count}/{total_seen}"
        })

        # LOGGING
        actual_step = step // ppo_config.batch_size

        if actual_step > 0 and actual_step % LOG_EVERY == 0:
            tqdm.write("\n---------------------------")
            tqdm.write(f"Epoch {epoch} Step {actual_step}")
            tqdm.write(f"Avg Reward: {round(avg,3)}")

            idx = random.randint(0, len(batch_responses_text)-1)
            tqdm.write(f"Q: {batch_meta[idx][0]}")
            tqdm.write(f"SQL: {batch_responses_text[idx]}")
            tqdm.write(f"Reward: {round(batch_rewards[idx].item(),3)}")

    # SAVE BEST
    if avg > best_reward:
        best_reward = avg
        path = os.path.join(PROJECT_ROOT, "checkpoints/best_rlhf_codet5_soft")
        os.makedirs(path, exist_ok=True)

        model.save_pretrained(path)
        tokenizer.save_pretrained(path)

        tqdm.write(f"\n✅ BEST MODEL SAVED → {path}")
