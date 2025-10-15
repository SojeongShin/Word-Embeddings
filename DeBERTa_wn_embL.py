import torch
from transformers import AutoModel, AutoTokenizer
from nltk.corpus import wordnet as wn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # =============================
# # 0. 모델과 토크나이저 불러오기
# # =============================
# model_name = "microsoft/deberta-v3-large"   # BERT/RoBERTa로 바꿔도 가능
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModel.from_pretrained(model_name).to(device)

# # embedding layer (연결된 상태)
# embedding_layer = model.embeddings.word_embeddings  # nn.Embedding

# =============================
# 1. Base embedding 추출
# =============================
def get_base_embedding(word, tokenizer, embedding_layer, normalize=True):
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
            "sense_id": s.name(), "pos": s.pos(), "gloss": s.definition(),
        })

    return {
        "base_embedding": base_emb,
        "senses": senses
    }


def text_to_emb(text, tokenizer, embedding_layer, device, agg="mean"):
    toks = tokenizer.tokenize(text)
    
    if not toks:
        return None
    
    ids = torch.tensor(tokenizer.convert_tokens_to_ids(toks), device=device)
    embs = embedding_layer(ids)
    
    return embs.mean(dim=0) if agg == "mean" else embs.sum(dim=0)


# =============================
# 3. Sense embedding 초기화
# =============================
def init_sense_embeddings(
    inventory, tokenizer, embedding_layer, device,
    use_examples=False, include_lemmas=False, include_hypernyms=False,
    alpha=0.0, normalize=True
):
    base_emb = inventory["base_embedding"]
    sense_embs = {}
    
    for s in inventory["senses"]:
        syn = wn.synset(s["sense_id"])
        parts = [syn.definition()]
        if use_examples and syn.examples(): parts += syn.examples()
        
        if include_lemmas: parts += [l.name().replace('_', ' ') for l in syn.lemmas()]

        if include_hypernyms:
            for h in syn.hypernyms():
                parts.append(h.definition())
                
        sense_text = ". ".join([p for p in parts if p and p.strip()])
        v = text_to_emb(sense_text, tokenizer, embedding_layer, device)

        if v is None or v.numel() == 0:
            v = base_emb.clone()

        v = (1 - alpha) * v + alpha * base_emb
        
        if normalize: v = v / (v.norm(p=2) + 1e-12)
        sense_embs[s["sense_id"]] = v
    return sense_embs


# =============================
# 4. 정의문 기반 업데이트 (식 1)
# =============================
def definition_update(sense_id, sense_embs, tokenizer, embedding_layer, beta):
    gloss = wn.synset(sense_id).definition()
    gloss_emb = text_to_emb(gloss, tokenizer, embedding_layer, device)
    
    if gloss_emb is None:
        return sense_embs[sense_id]
    
    v_old = sense_embs[sense_id]
    
    return (1 - beta) * v_old + beta * gloss_emb


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
