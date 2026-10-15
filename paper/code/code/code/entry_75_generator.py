# ════════════════════════════════════════════════════════════════
# COMPRENEUR — Entry 75: The Verified Generator (v1) — The Five Attractors
# Result: prediction-bras 0.500 canon / 0.000 perm; verdict-bras dead at
#   arithmetically-closed degenerate optima. Seal d9c1ebc576.
# Run: PyTorch, T4 ~5 min (or slow CPU).
# ════════════════════════════════════════════════════════════════
import torch, torch.nn as nn, torch.nn.functional as F
import random, time, hashlib
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
t0 = time.time(); SEED = 75

ACTANTS = ["marie","paul","chien","chat","grifflor","blorpte","frumilax",
           "zornik","purbet","chalimbe","vortaxe","snellor"]
OBJETS  = ["livre","pomme","velo","ballon","cle","lampe","stylo","souris"]
LIEUX   = ["salle","cave","grenier"]
ACTIONS = ["possede","habite","mange"]
NOMS = ACTANTS + OBJETS + LIEUX
DETS = ["le","la"]
GENRE = {w: ("le" if w in ("chien","chat","zornik","purbet",
                           "livre","velo","stylo","grenier") else "la")
         for w in NOMS}
MOTS = NOMS + ACTIONS + DETS
SPECIALS = ["<s>","<sep>","</s>","TAG_S","TAG_A","TAG_O"]
TOKS = MOTS + SPECIALS ; V = len(TOKS)
IDX = {t: i for i, t in enumerate(TOKS)}
BOS, SEP, EOS = IDX["<s>"], IDX["<sep>"], IDX["</s>"]
TAGS = {"S": IDX["TAG_S"], "A": IDX["TAG_A"], "O": IDX["TAG_O"]}
INTERDITS = [IDX[t] for t in SPECIALS if t != "</s>"]

FAITS_CANON = []
for i, a in enumerate(ACTANTS):
    FAITS_CANON.append((a, "possede", OBJETS[i % 8]))
    FAITS_CANON.append((a, "habite",  LIEUX[i % 3]))
FAITS_CANON += [(a, "mange", ["pomme","souris"][i % 2])
                for i, a in enumerate(ACTANTS[:6])]
FAITS_PERM = [(o, "possede", ACTANTS[(i*5+2) % 12]) for i, o in enumerate(OBJETS)]
FAITS_PERM += [(l, "possede", ACTANTS[(i*4+1) % 12]) for i, l in enumerate(LIEUX)]
FAITS_PERM += [("livre","mange","marie"), ("pomme","habite","paul")]
FAITS_EQUIL = FAITS_CANON + FAITS_PERM

r2 = random.Random(750); used = set(FAITS_EQUIL)
SECRETS_C = []
while len(SECRETS_C) < 16:
    a = r2.choice(ACTANTS); act = r2.choice(ACTIONS)
    o = r2.choice(LIEUX if act == "habite" else OBJETS)
    if (a, act, o) not in used and (a, act, o) not in SECRETS_C:
        SECRETS_C.append((a, act, o))
SECRETS_P = []
while len(SECRETS_P) < 16:
    s = r2.choice(OBJETS + LIEUX); act = r2.choice(ACTIONS)
    o = r2.choice(ACTANTS)
    if (s, act, o) not in used and (s, act, o) not in SECRETS_P:
        SECRETS_P.append((s, act, o))
EVAL_CURVE = SECRETS_C[:12]

def prefix(f, r):
    s, a, o = f
    paires = [(TAGS["S"], IDX[s]), (TAGS["A"], IDX[a]), (TAGS["O"], IDX[o])]
    r.shuffle(paires)
    out = [BOS]
    for tag, w in paires: out += [tag, w]
    return out + [SEP]

def reference(f):
    s, a, o = f
    return [GENRE[s], s, a, GENRE[o], o]

def juger(emis, f, strict=True):
    s, a, o = f
    if not (1 <= len(emis) <= 7):
        return 0.0, False, (0.0, 0.0, 0.0, 0.0)
    core = [w for w in emis if w not in DETS]
    act = next((w for w in core if w in ACTIONS), None)
    if act is None:
        return 0.0, False, (0.0, 0.0, 0.0, 0.0)
    i = core.index(act)
    c_a = float(act == a)
    c_s = float(core[:i] == [s])
    c_o = float(core[i+1:] == [o])
    c_d = 1.0
    for k, w in enumerate(emis):
        if w in NOMS and (k == 0 or emis[k-1] != GENRE[w]):
            c_d = 0.0
    if strict:
        return (c_a+c_s+c_o+c_d)/4, (c_a+c_s+c_o+c_d)==4, (c_a,c_s,c_o,c_d)
    return (c_a+c_s+c_o)/3, (c_a+c_s+c_o)==3, (c_a,c_s,c_o,c_d)

class Modele(nn.Module):
    def __init__(self, d=128, têtes=4, couches=2):
        super().__init__()
        self.emb = nn.Embedding(V, d)
        self.pos = nn.Parameter(torch.randn(1, 24, d) * 0.02)
        bloc = nn.TransformerEncoderLayer(d, têtes, d*4, dropout=0.0,
                                          activation="gelu", batch_first=True)
        self.enc = nn.TransformerEncoder(bloc, couches,
                                         enable_nested_tensor=False)
        self.tete = nn.Linear(d, V)
    def forward(self, x):
        T = x.size(1)
        masque = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), 1)
        return self.tete(self.enc(self.emb(x) + self.pos[:, :T], mask=masque))

def generer(m, prefs, maxlen=7, greedy=False):
    B = prefs.size(0)
    seq = prefs ; done = torch.zeros(B, dtype=torch.bool, device=device)
    logp_tot = torch.zeros(B, device=device)
    for _ in range(maxlen):
        logits = m(seq)[:, -1, :]
        logits[:, INTERDITS] = -1e9
        logp = F.log_softmax(logits, -1)
        tok = logp.argmax(-1) if greedy else torch.multinomial(logp.exp(), 1).squeeze(1)
        tok = torch.where(done, torch.full_like(tok, EOS), tok)
        pas = logp.gather(1, tok.unsqueeze(1)).squeeze(1)
        logp_tot = logp_tot + torch.where(done, torch.zeros_like(pas), pas)
        seq = torch.cat([seq, tok.unsqueeze(1)], 1)
        done = done | (tok == EOS)
        if done.all(): break
    emis = []
    for b in range(B):
        mots = []
        for j in range(8, seq.size(1)):
            tk = seq[b, j].item()
            if tk == EOS: break
            mots.append(TOKS[tk])
        emis.append(mots)
    return logp_tot, emis

def evaluer(m, faits, B=32):
    rr = random.Random(4242)
    st = mn = 0 ; det_n = det_ok = 0
    for k in range(0, len(faits), B):
        chunk = faits[k:k+B]
        prefs = torch.tensor([prefix(f, rr) for f in chunk], device=device)
        with torch.no_grad():
            _, emis = generer(m, prefs, greedy=True)
        for e, f in zip(emis, chunk):
            st += juger(e, f, True)[1] ; mn += juger(e, f, False)[1]
            for j, w in enumerate(e):
                if w in NOMS:
                    det_n += 1
                    if j > 0 and e[j-1] == GENRE[w]: det_ok += 1
    n = len(faits)
    return st/n, mn/n, (det_ok/det_n if det_n else float("nan")), det_n

def entrainer_P(faits, tag, updates=2000, B=64, lr=3e-4):
    torch.manual_seed(SEED + 3)
    m = Modele().to(device) ; opt = torch.optim.AdamW(m.parameters(), lr=lr)
    r = random.Random(SEED + 3)
    for u in range(updates):
        batch = [faits[r.randrange(len(faits))] for _ in range(B)]
        prefs = torch.tensor([prefix(f, r) for f in batch], device=device)
        refs = torch.tensor([[IDX[w] for w in reference(f)] + [EOS]
                             for f in batch], device=device)
        seq = torch.cat([prefs, refs], 1)
        logits = m(seq)[:, 7:13, :]
        logits[:, :, INTERDITS] = -1e9
        loss = F.cross_entropy(logits.reshape(-1, V), refs.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    return m

def entrainer_V(faits, tag, strict=True, updates=3500, B=48, lr=3e-4):
    torch.manual_seed(SEED + (1 if strict else 2))
    m = Modele().to(device) ; opt = torch.optim.AdamW(m.parameters(), lr=lr)
    r = random.Random(SEED + (1 if strict else 2))
    for u in range(updates):
        batch = [faits[r.randrange(len(faits))] for _ in range(B)]
        prefs = torch.tensor([prefix(f, r) for f in batch], device=device)
        logp, emis = generer(m, prefs)
        R = torch.tensor([juger(e, f, strict)[0]
                          for e, f in zip(emis, batch)], device=device)
        loss = -((R - R.mean()) * logp).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    return m

if __name__ == "__main__":
    print(f"Unit: {device}")
    print("── P-canon (imitation) ──")
    mPc = entrainer_P(FAITS_CANON, "P-canon", 2000)
    print("── V (REINFORCE on verdict) ──")
    mVc = entrainer_V(FAITS_EQUIL, "V", True, 3500)
    print("\n═══ VERDICT 75 ═══")
    for nom, m in [("P-canon", mPc), ("V", mVc)]:
        sc = evaluer(m, SECRETS_C) ; sp = evaluer(m, SECRETS_P)
        print(f"  {nom:<10} S-canon {sc[0]:.3f} · S-perm {sp[0]:.3f} · struct {sc[1]:.3f}")
    print(f"\n  The attractors: V dies at its degenerate optimum "
          f"(see paper 4-bis.2). Seal d9c1ebc576 · {time.time()-t0:.0f} s")
