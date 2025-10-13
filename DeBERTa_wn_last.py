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
# 1. Base embedding 추출 (last 4 layers 평균, 특수토큰 제외)
# =============================
def get_base_embedding(word, tokenizer, model, device, max_length=16):
    model.eval()
    inputs = tokenizer(word, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    last4 = torch.stack(out.hidden_states[-4:], dim=0).mean(dim=0)  # (1,L,H)
    ids = inputs["input_ids"][0]
    specials = {tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id}
    mask = torch.tensor([tid not in specials for tid in ids], device=device)
    v = last4[0][mask].mean(dim=0) if mask.any() else last4.mean(dim=1)[0]
    v = v / (v.norm(p=2) + 1e-12)
    return v


# =============================
# 2. Sense inventory 구축
# =============================
def build_sense_inventory(word, tokenizer, model, device):
    synsets = wn.synsets(word)
    if not synsets:
        return None
    base_emb = get_base_embedding(word, tokenizer, model, device)
    if base_emb is None:
        return None
    senses = [{"sense_id": s.name(), "pos": s.pos(), "gloss": s.definition()} for s in synsets]
    return {"base_embedding": base_emb, "senses": senses}



def text_to_emb(text, tokenizer, model, device, max_length=128):
    model.eval()
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=max_length, add_special_tokens=True
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    # hidden_states: tuple(len=layers+1), each (B, L, H)
    last4 = outputs.hidden_states[-4:]                   # tuple of 4 tensors
    stacked = torch.stack(last4, dim=0).mean(dim=0)      # (B, L, H), 평균(또는 가중합) 가능

    # 특수 토큰 제외 평균
    input_ids = inputs["input_ids"][0]                   # (L,)
    special_ids = {tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id}
    mask = torch.tensor([tid not in special_ids for tid in input_ids], device=device)  # (L,)
    if mask.sum() == 0:
        v = stacked.mean(dim=1)[0]       # 모든 토큰 평균(폴백)
    else:
        v = stacked[0][mask].mean(dim=0) # 특수토큰 제외 평균
    v = v / (v.norm(p=2) + 1e-12)
    return v




# =============================
# 3. Sense embedding 초기화
# =============================
def init_sense_embeddings(
    inventory, tokenizer, model, device,
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
        v = text_to_emb(sense_text, tokenizer, model, device)

        if v is None or v.numel() == 0:
            v = base_emb.clone()

        v = (1 - alpha) * v + alpha * base_emb
        
        if normalize: v = v / (v.norm(p=2) + 1e-12)
        sense_embs[s["sense_id"]] = v
    return sense_embs


# =============================
# 4. 정의문 기반 업데이트 (식 1)
# =============================
def definition_update(sense_id, sense_embs, tokenizer, model, device, beta):
    gloss = wn.synset(sense_id).definition()
    gloss_emb = text_to_emb(gloss, tokenizer, model, device)

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
def train_one_epoch(inventory, sense_embs, tokenizer, model, device, beta, gamma):

    base_emb = inventory["base_embedding"]
    senses = inventory["senses"]
    r_js = [definition_update(s["sense_id"], sense_embs, tokenizer, model, device, beta) for s in senses]

    if not r_js: return sense_embs
    v_js = base_alignment(r_js, base_emb, gamma)
    
    for s, v in zip(senses, v_js):
        sense_embs[s["sense_id"]] = v
    return sense_embs
