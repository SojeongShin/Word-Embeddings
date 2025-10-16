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
from DeBERTa_wn_baseinit import get_base_embedding, build_sense_inventory, init_sense_embeddings, train_one_epoch

# =============================
# 0. 준비
# =============================
import random
import numpy as np

import time
start_time = time.time()  # 실행 시작 시각

OUTPUT_DIR = os.path.join("results/embedding-layer/b=0.25_g=0.7/len64", "noise_base_init")  # 결과 저장 디렉토리
os.makedirs(OUTPUT_DIR, exist_ok=True)


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
dim = model.config.hidden_size
embedding_layer=model.embeddings.word_embeddings
print(f"[Model] Loaded {MODEL_NAME} with hidden size {dim}")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Device] Using {device}")
model.to(device)
model.eval()

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
    init_embs = init_sense_embeddings(inv, dim)
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

# Early Stopping 설정
patience = 3        
min_delta = 1e-4     # 최소 개선폭
best_corr = -float("inf")
no_improve = 0
early_stop = False   # 플래그

for epoch in range(num_epochs):
    if early_stop:
        print(f"[EarlyStop] Triggered early stop at epoch {epoch}.")
        break

    print(f"\n[Epoch {epoch+1}/{num_epochs}]")
    for w, inv in inventory.items():
        sense_embs = train_one_epoch(
            inv, sense_embs, tokenizer, embedding_layer, 
            beta=0.25,   # gloss 반영 정도
            gamma=0.7    # base 정렬 정도
        )
    
    corr = evaluate_simlex(sense_embs, inventory, path="SimLex-999.txt", max_pairs=None)
    epoch_corrs.append(corr)

    # Early Stopping 로직
    if corr is None:
        current = -float("inf")
    else:
        current = corr

    if current - best_corr > min_delta:
        best_corr = current
        no_improve = 0
        print(f"[EarlyStop] Improvement detected. best_corr={best_corr:.6f}")
    else:
        no_improve += 1
        print(f"[EarlyStop] No significant improvement ({no_improve}/{patience}).")

        if no_improve >= patience:
            print(f"[EarlyStop] Stop criterion met (no improvement > {min_delta} for {patience} epochs).")
            early_stop = True  # early stop 플래그

# =============================
# 7. 시각화 및 저장 (early stop 대응)
# =============================
import time

end_time = time.time()
elapsed = end_time - start_time

# x축은 실제 수집된 성능 길이에 맞추기
x_epochs = list(range(1, len(epoch_corrs) + 1))

plt.figure(figsize=(8,5))
plt.plot(x_epochs, epoch_corrs, marker="o", linestyle="-", label="Sense Embeddings")

# baseline이 있을 때만 표시
if baseline_results:
    plt.axhline(y=baseline_corr, color="red", linestyle="--",
                label=f"Baseline (r={baseline_corr:.3f})")

# 조기 종료 시점 수직선(선택)
if len(epoch_corrs) < num_epochs:
    plt.axvline(x=len(epoch_corrs), linestyle=":", color="gray",
                label=f"Early stop @ {len(epoch_corrs)}")

plt.xlabel("Epoch")
plt.ylabel("Pearson Correlation")
final_r = epoch_corrs[-1] if epoch_corrs else float("nan")
plt.title(f"Pearson Correlation on SimLex-999 (final r={final_r:.3f}, epochs={len(epoch_corrs)})")
plt.legend()
plt.grid(True)

# 저장 경로 (기존 OUTPUT_DIR 사용)
timestamp = time.strftime("%Y%m%d-%H%M%S")
fig_path = os.path.join(OUTPUT_DIR, f"simlex_corr_{MODEL_NAME.replace('/','_')}_{timestamp}.png")
plt.savefig(fig_path, dpi=200, bbox_inches="tight")
print(f"[Saved] Figure saved to: {fig_path}")

# 전체 경과 시간 출력
h = int(elapsed // 3600); m = int((elapsed % 3600) // 60); s = int(elapsed % 60)
print(f"[Time] Total elapsed: {h:02d}:{m:02d}:{s:02d} ({elapsed:.2f}s)")

from DeBERTa_wn_mid import TOTAL_TOKEN_COUNT
print(f"\n[Summary] Total tokens processed during initialization + training: {TOTAL_TOKEN_COUNT}")