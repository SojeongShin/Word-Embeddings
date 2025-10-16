import torch
from transformers import AutoModel, AutoTokenizer
from nltk.corpus import wordnet as wn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 64
AGG = "mean"
TOTAL_TOKEN_COUNT = 0

# =============================
# 1. Base embedding 추출
# =============================
def get_base_embedding(word, tokenizer, embedding_layer, normalize=True):
    global TOTAL_TOKEN_COUNT

    tokens = tokenizer.tokenize(word)
    if not tokens:
        return None
    
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    token_ids = torch.tensor(token_ids, device=device)
    sub_embs = embedding_layer(token_ids)  # nn.Embedding forward pass

    emb = sub_embs.mean(dim=0)

    if normalize:
        emb = emb / (emb.norm(p=2) + 1e-12)
    return emb

# =============================
# 2. Sense inventory 구축
# =============================
def build_sense_inventory(word, tokenizer, embedding_layer):
    synsets = wn.synsets(word)
    if not synsets:
        return None

    base_emb = get_base_embedding(word, tokenizer, embedding_layer)
    if base_emb is None:
        return None

    senses = []
    for s in synsets:
        senses.append({
            "sense_id": s.name(), "pos": s.pos(), "gloss": s.definition(),})
    return { "base_embedding": base_emb, "senses": senses }


def text_to_emb(text, tokenizer, embedding_layer, device, agg=AGG):
    toks = tokenizer.tokenize(text)
    if not toks:
        return None
    
    # 길이 고정
    toks = toks[:MAX_LEN]

    ids = torch.tensor(tokenizer.convert_tokens_to_ids(toks), device=device)
    embs = embedding_layer(ids)
    v = embs.mean(dim=0) if agg == AGG else embs.sum(dim=0)

    # sense L2 정규화
    v = v / (v.norm(p=2) + 1e-12)
    return v


# =============================
# 3. Sense embedding 초기화
# =============================
def init_sense_embeddings(inventory, dim):
    sense_embs = {}
    base_emb = inventory["base_embedding"]  # 이미 L2 정규화됨

    for s in inventory["senses"]:
        sid = s["sense_id"]
        v = base_emb + 0.01 * torch.randn(dim, device=device)
        v = v / (v.norm(p=2) + 1e-12)
        sense_embs[sid] = v
    return sense_embs


# =============================
# 4. 정의문 기반 업데이트 (식 1)
# =============================
def definition_update(sense_id, sense_embs, tokenizer, embedding_layer, beta):
    gloss = wn.synset(sense_id).definition()
    gloss_emb = text_to_emb(gloss, tokenizer, embedding_layer, device)  # 이미 L2 정규화 반환
    if gloss_emb is None:
        return sense_embs[sense_id]

    v_old = sense_embs[sense_id]
    v_new = (1 - beta) * v_old + beta * gloss_emb
    v_new = v_new / (v_new.norm(p=2) + 1e-12) 
    return v_new



# =============================
# 5. Base alignment (식 4)
# =============================
def base_alignment(r_js, base_emb, gamma):
    r_stack = torch.stack(r_js)
    r_bar = r_stack.mean(dim=0)
    
    return [r_j + (1 - gamma) * (base_emb - r_bar) for r_j in r_js]

# =============================
# 6. 학습 루프 (1 epoch)
# =============================
def train_one_epoch(inventory, sense_embs, tokenizer, embedding_layer, beta=0.25, gamma=0.7):
    
    base_emb = inventory["base_embedding"]
    senses = inventory["senses"]
    r_js = [definition_update(s["sense_id"], sense_embs, tokenizer, embedding_layer, beta) for s in senses]
    
    if not r_js: return sense_embs
    v_js = base_alignment(r_js, base_emb, gamma)
    
    for s, v in zip(senses, v_js):
        sense_embs[s["sense_id"]] = v
    return sense_embs
