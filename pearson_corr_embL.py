import os
import pandas as pd
from scipy.stats import pearsonr
import nltk
import torch.nn.functional as F
from nltk.corpus import wordnet as wn
from transformers import AutoTokenizer, AutoModel
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.animation import FuncAnimation, writers
import csv
from DeBERTa_wn_embL import get_base_embedding, build_sense_inventory, init_sense_embeddings, train_one_epoch

# =============================
# 0. 준비
# =============================
import random
import numpy as np

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)



nltk.download("wordnet")
nltk.download("omw-1.4")

MODEL_NAME = "microsoft/deberta-v3-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

# embedding layer (연결된 상태)
embedding_layer = model.embeddings.word_embeddings
dim = embedding_layer.embedding_dim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Device] Using {device}")
model.to(device)


# =============================
# 1. 벤치마크 로드 (SimLex-999)
# =============================
def load_simlex999(path="SimLex-999.txt"):
    df = pd.read_csv(path, sep="\t")
    return df[["word1", "word2", "SimLex999"]]

# =============================
# 2. 두 단어의 유사도 계산
# =============================
def word_similarity(word1, word2, sense_embs, inventory):
    if word1 not in inventory or word2 not in inventory:
        return None

    senses1 = inventory[word1]["senses"]
    senses2 = inventory[word2]["senses"]

    emb1 = [sense_embs[s["sense_id"]] for s in senses1 if s["sense_id"] in sense_embs]
    emb2 = [sense_embs[s["sense_id"]] for s in senses2 if s["sense_id"] in sense_embs]

    if not emb1 or not emb2:
        return None

    sims = []
    for e1 in emb1:
        for e2 in emb2:
            cos_sim = F.cosine_similarity(e1.unsqueeze(0), e2.unsqueeze(0)).item()
            sims.append(cos_sim)

    return max(sims) if sims else None

# =============================
# 3. SimLex-999 평가
# =============================
def evaluate_simlex(sense_embs, inventory, path="SimLex-999.txt", max_pairs=None):
    df = load_simlex999(path)

    results = []
    for i, row in df.iterrows():
        if max_pairs and i >= max_pairs:
            break
        w1, w2, gold = row["word1"], row["word2"], row["SimLex999"]
        sim = word_similarity(w1, w2, sense_embs, inventory)
        if sim is not None:
            results.append({
                "word1": w1,
                "word2": w2,
                "gold_score": gold,
                "model_score": sim
            })

    if not results:
        print("평가할 수 있는 단어 쌍이 없습니다.")
        return None

    gold_scores = [r["gold_score"] for r in results]
    model_scores = [r["model_score"] for r in results]
    corr, _ = pearsonr(model_scores, gold_scores)
    print(f"[Evaluation] Pearson correlation on SimLex-999: {corr:.4f} (n={len(results)})")

    return corr

# =============================
# 4. Inventory 구축
# =============================
simlex = load_simlex999("SimLex-999.txt")
all_words = set(simlex["word1"]).union(set(simlex["word2"]))

inventory = {}
for w in all_words:
    inv = build_sense_inventory(w, tokenizer, embedding_layer)
    if inv:
        inventory[w] = inv

sense_embs = {}
for w, inv in inventory.items():
    init_embs = init_sense_embeddings(
        inv,               # inventory
        tokenizer,
        embedding_layer,
        device,
        # 필요 시 조절
        use_examples=False,
        include_lemmas=False,
        include_hypernyms=False,
        alpha=0.0,
        normalize=True     # L2 정규화
    )
    sense_embs.update(init_embs)


# =============================
# 5. Baseline Pearson correlation
# =============================
baseline_results = []
for _, row in simlex.iterrows():
    w1, w2, gold = row["word1"], row["word2"], row["SimLex999"]

    emb1 = get_base_embedding(w1, tokenizer, embedding_layer)
    emb2 = get_base_embedding(w2, tokenizer, embedding_layer)

    if emb1 is None or emb2 is None:
        continue

    sim = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
    baseline_results.append((gold, sim))

if baseline_results:
    gold_scores = [g for g, _ in baseline_results]
    model_scores = [s for _, s in baseline_results]
    baseline_corr, _ = pearsonr(model_scores, gold_scores)
    print(f"[Baseline] Pearson correlation: {baseline_corr:.4f}")
else:
    baseline_corr = 0.0
    print("Baseline 평가 불가")

# =============================
# 6. Sense Embedding 학습 루프
# =============================
num_epochs = 30
epoch_corrs = []

for epoch in range(num_epochs):
    print(f"\n[Epoch {epoch+1}/{num_epochs}]")
    for w, inv in inventory.items():
        sense_embs = train_one_epoch(inv, sense_embs, tokenizer, embedding_layer)
    corr = evaluate_simlex(sense_embs, inventory, path="SimLex-999.txt", max_pairs=None)
    epoch_corrs.append(corr)

# =============================
# 7. 시각화 및 저장
# =============================
import os

SAVE_DIR = "results/embedding-layer/b=0.25_g=0.7"  # 저장할 폴더 경로
os.makedirs(SAVE_DIR, exist_ok=True)

save_path = os.path.join(SAVE_DIR, "gloss_init_simlex.png")

plt.figure(figsize=(8,5))
plt.plot(range(1, num_epochs+1), epoch_corrs, marker="o", linestyle="-", color="blue", label="Sense Embeddings")
plt.axhline(y=baseline_corr, color="red", linestyle="--", label=f"Baseline (r={baseline_corr:.3f})")
plt.xlabel("Epoch")
plt.ylabel("Pearson Correlation (SimLex-999)")
plt.title("Epoch vs Pearson Correlation on SimLex-999")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(save_path, dpi=300)  
print(f"[Saved Plot] {save_path}")
