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
from DeBERTa_wordnet_transformer import get_base_embedding, build_sense_inventory, init_sense_embeddings, train_one_epoch

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

    # out_file = "simlex_eval_results.csv"
    # with open(out_file, "w", newline="", encoding="utf-8") as f:
    #     writer = csv.DictWriter(f, fieldnames=["word1", "word2", "gold_score", "model_score"])
    #     writer.writeheader()
    #     writer.writerows(results)

    # print(f"[Saved] Detailed results saved to {out_file}")
    return corr

# =============================
# 4. Inventory 구축
# =============================
simlex = load_simlex999("SimLex-999.txt")
all_words = set(simlex["word1"]).union(set(simlex["word2"]))

inventory = {}
for w in all_words:
    inv = build_sense_inventory(w, tokenizer, model, device)
    if inv:
        inventory[w] = inv


sense_embs = {}
for w, inv in inventory.items():
    init_embs = init_sense_embeddings(
        inv, tokenizer, model, device,
        use_examples=False,     # gloss만 사용 (원하면 True)
        include_lemmas=False,   # lemma도 섞고 싶으면 True
        include_hypernyms=False,# 상위개념 정의까지 넣고 싶으면 True
        alpha=0.0,              # base 앵커링 비율(0=순수 gloss)
        normalize=True          # L2 정규화
    )
    sense_embs.update(init_embs)


# =============================
# 5. Baseline Pearson correlation
# =============================
baseline_results = []
for _, row in simlex.iterrows():
    w1, w2, gold = row["word1"], row["word2"], row["SimLex999"]

    emb1 = get_base_embedding(w1, tokenizer, model, device)
    emb2 = get_base_embedding(w2, tokenizer, model, device)

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
        sense_embs = train_one_epoch(
            inv, sense_embs, tokenizer, model, device, 
            beta=0.25,   # gloss 반영 정도
            gamma=0.7    # base 정렬 정도
        )
    corr = evaluate_simlex(sense_embs, inventory, path="SimLex-999.txt", max_pairs=None)
    epoch_corrs.append(corr)

# # =============================
# # 7. 시각화
# # =============================
plt.figure(figsize=(8,5))
plt.plot(range(1, num_epochs+1), epoch_corrs, marker="o", linestyle="-", color="blue", label="Sense Embeddings")
plt.axhline(y=baseline_corr, color="red", linestyle="--", label=f"Baseline (r={baseline_corr:.3f})")
plt.xlabel("Epoch")
plt.ylabel("Pearson Correlation")
plt.title("Pearson Correlation on SimLex-999 (encoder-based sense init)")
plt.legend()
plt.grid(True)
plt.show()


# # =============================
# # 동영상 시각화
# # =============================
# def make_sense_evolution_video(words, inventory, sense_embs, tokenizer, embedding_layer,
#                                num_epochs=10, out_video="sense_alignment.mp4"):
#     fig, ax = plt.subplots(figsize=(8,6))

#     def update(epoch):
#         ax.clear()
#         ax.set_title(f"Word-Sense Embedding Distribution (Epoch {epoch+1})")
#         all_vecs, labels, colors = [], [], []
#         for w, inv in inventory.items():
#             base = inv["base_embedding"].detach().cpu().numpy()
#             all_vecs.append(base)
#             labels.append(f"{w}_BASE")
#             colors.append("red")
#             for s in inv["senses"]:
#                 sid = s["sense_id"]
#                 if sid in sense_embs:
#                     v = sense_embs[sid].detach().cpu().numpy()
#                     all_vecs.append(v)
#                     labels.append(f"{w}_{sid}")
#                     colors.append("blue")

#         if len(all_vecs) < 2:
#             return

#         reduced = PCA(n_components=2).fit_transform(all_vecs)

#         for i, label in enumerate(labels):
#             if label.endswith("BASE"):
#                 ax.scatter(reduced[i,0], reduced[i,1], marker="*", s=200, c=colors[i])
#             else:
#                 ax.scatter(reduced[i,0], reduced[i,1], marker="o", s=50, c=colors[i])
#             ax.text(reduced[i,0]+0.01, reduced[i,1]+0.01, label, fontsize=6)
#         ax.grid(True)

#         # 한 epoch 학습
#         for w, inv in inventory.items():
#             train_one_epoch(inv, sense_embs, tokenizer, embedding_layer)

#     ani = FuncAnimation(fig, update, frames=num_epochs, interval=1000, repeat=False)

#     Writer = writers['ffmpeg']
#     writer = Writer(fps=1, metadata=dict(artist='Me'), bitrate=1800)
#     ani.save(out_video, writer=writer)
#     print(f"[Saved] Video saved to {out_video}")

# # =============================
# # 실행 예시
# # =============================
# words_sample = ["bank"]

# inventory = {}
# for w in words_sample:
#     inv = build_sense_inventory(w, tokenizer, embedding_layer)
#     if inv:
#         inventory[w] = inv

# sense_embs = {}
# for w, inv in inventory.items():
#     sense_embs.update(init_sense_embeddings(inv, dim))

# make_sense_evolution_video(words_sample, inventory, sense_embs,
#                            tokenizer, embedding_layer,
#                            num_epochs=10, out_video="sense_alignment.mp4")