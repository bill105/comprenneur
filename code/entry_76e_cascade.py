# ════════════════════════════════════════════════════════════════
# COMPRENEUR — Entry 76-e: The Escape Cascade (frontier distillation)
# Result: canon greedy k5 0.69, DET agreement 97% (43-fact world).
# Seal d9c1ebc576. Run: PyTorch, T4 ~12 min.
# ════════════════════════════════════════════════════════════════
import torch, torch.nn as nn, torch.nn.functional as F
import random, time, hashlib
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
t0 = time.time(); SEED = 76
TEMP_BASE, EPS_BASE = 1.3, 0.05

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
AUTORISES = torch.zeros(V, device=device) ; AUTORISES[:] = True
AUTORISES[INTERDITS] = False

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

def parfaite(f):
    s, a, o = f
    return [GENRE[s], s, a, GENRE[o], o]

def marche_k(emis, f):
    s, a, o = f
    if not emis: return 0
    k = 0
    if a in emis: k = 1
    if k and s in emis and o in emis: k = 2
    core = [w for w in emis if w not in DETS]
    if k == 2 and core == [s, a, o]:
        k = 3
        i_s, i_o = emis.index(s), emis.index(o)
        if i_s > 0 and emis[i_s-1] in DETS and i_o > 0 and emis[i_o-1] in DETS:
            k = 4
            if emis == parfaite(f): k = 5
    return k

class Modele(nn.Module):
    def __init__(self, d=128, têtes=4, couches=2):
        super().__init__()
        self.emb = nn.Embedding(V, d)
        self.pos = nn.Parameter(torch.randn(1, 24, d) * 0.02)
        bloc = nn.TransformerEncoderLayer(d, têtes, d*4, dropout=0.0,
                                          activation="gelu", batch_first=True)
        self.enc = nn.TransformerEncoder(bloc, couches, enable_nested_tensor=False)
        self.tete = nn.Linear(d, V)
    def forward(self, x):
        T = x.size(1)
        masque = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), 1)
        return self.tete(self.enc(self.emb(x) + self.pos[:, :T], mask=masque))

def _boucle(m, prefs, maxlen=7, greedy=False, temp=TEMP_BASE, eps=EPS_BASE):
    B = prefs.size(0)
    seq = prefs ; done = torch.zeros(B, dtype=torch.bool, device=device)
    for _ in range(maxlen):
        logits = m(seq)[:, -1, :].clone()
        logits[:, INTERDITS] = -1e9
        p = F.softmax(logits / temp, -1)
        p = (1 - eps) * p + eps * AUTORISES / AUTORISES.sum()
        tok = p.argmax(-1) if greedy else torch.multinomial(p, 1).squeeze(1)
        tok = torch.where(done, torch.full_like(tok, EOS), tok)
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
    return emis

def generer(m, prefs, maxlen=7, greedy=False, temp=TEMP_BASE, eps=EPS_BASE):
    return _boucle(m, prefs, maxlen, greedy, temp, eps)

def evaluer(m, faits, N=8, B=32):
    rr = random.Random(4242)
    g5 = 0 ; pol3 = pol5 = 0 ; det_n = det_ok = 0 ; nf = 0
    for k0 in range(0, len(faits), B):
        chunk = faits[k0:k0+B]
        prefs = torch.tensor([prefix(f, rr) for f in chunk], device=device)
        with torch.no_grad():
            eg = generer(m, prefs, greedy=True)
            eN = generer(m, prefs.repeat_interleave(N, 0))
        for i, f in enumerate(chunk):
            g5 += marche_k(eg[i], f) == 5
            pol3 += sum(marche_k(eN[N*i+t], f) >= 3 for t in range(N))
            pol5 += sum(marche_k(eN[N*i+t], f) == 5 for t in range(N))
            for j, w in enumerate(eg[i]):
                if w in NOMS:
                    det_n += 1
                    if j > 0 and eg[i][j-1] == GENRE[w]: det_ok += 1
            nf += 1
    return dict(g5=g5/nf, pol3=pol3/(N*nf), pol5=pol5/(N*nf),
                det=(det_ok/det_n if det_n else float("nan")))

def entrainer(faits, tag, graine, updates=2800, B=48, N_ech=10, lr=3e-4):
    torch.manual_seed(SEED + 20*graine)
    m = Modele().to(device) ; opt = torch.optim.AdamW(m.parameters(), lr=lr)
    r = random.Random(SEED + 20*graine)
    banque = {} ; premiere_k5 = None
    temp, eps = TEMP_BASE, EPS_BASE
    captures_recentes = 0 ; dernier_total = 0 ; pas_distill = 0
    for u in range(updates):
        batch = [faits[r.randrange(len(faits))] for _ in range(B)]
        prefs = torch.tensor([prefix(f, r) for f in batch], device=device)
        with torch.no_grad():
            emis = generer(m, prefs.repeat_interleave(N_ech, 0), temp=temp, eps=eps)
        for i, f in enumerate(batch):
            for t in range(N_ech):
                k = marche_k(emis[N_ech*i+t], f)
                if k > 0 and (f not in banque or k > banque[f][1]):
                    banque[f] = (emis[N_ech*i+t], k)
                    if k == 5 and premiere_k5 is None:
                        premiere_k5 = u
                        print(f"  [{tag}] first k5 (gender) captured at upd {u}")
        total = len(banque)
        captures_recentes += total - dernier_total ; dernier_total = total
        if (u+1) % 300 == 0:
            if captures_recentes < 3:
                temp = min(1.7, temp + 0.15) ; eps = min(0.15, eps + 0.02)
            else:
                temp = max(TEMP_BASE, temp - 0.05) ; eps = max(EPS_BASE, eps - 0.01)
            captures_recentes = 0
        if banque:
            par_niveau = {}
            for f, (ph, k) in banque.items():
                par_niveau.setdefault(k, []).append((f, ph))
            palier = []
            for k in sorted(par_niveau, reverse=True):
                palier += par_niveau[k]
                if len(palier) >= 3: break
            if len(palier) >= 3:
                idxs = random.choices(range(len(palier)), k=B)
                srcs, refs = [], []
                for j in idxs:
                    f, ph = palier[j]
                    srcs.append(prefix(f, r))
                    refs.append([IDX[w] for w in ph] + [EOS])
                L = max(len(x) for x in refs)
                src_t = torch.tensor(srcs, device=device)
                ref_t = torch.tensor([x + [EOS]*(L-len(x)) for x in refs], device=device)
                mask = (torch.arange(L, device=device)[None, :]
                        < torch.tensor([len(x) for x in refs], device=device)[:, None])
                seq = torch.cat([src_t, ref_t], 1)
                logits = m(seq)[:, 7:7+L, :]
                logits[:, :, INTERDITS] = -1e9
                perte = (F.cross_entropy(logits.reshape(-1, V),
                        ref_t.reshape(-1), reduction='none'
                        ).reshape(B, L) * mask).sum() / mask.sum()
                opt.zero_grad(); perte.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
                pas_distill += 1
    return m, banque, premiere_k5

if __name__ == "__main__":
    print(f"Unit: {device}")
    print("76-e: escape cascade (frontier distillation, tiered)\n")
    m1, ban1, pk1 = entrainer(FAITS_EQUIL, "g1", graine=1)
    sc = evaluer(m1, SECRETS_C, N=8)
    dd = f"{sc['det']:.0%}" if sc['det'] == sc['det'] else "—"
    print(f"\n  SECRETS CANON: order {sc['pol3']:.2f} · perfect {sc['pol5']:.2f}"
          f" · greedy k5 {sc['g5']:.2f} · DET {dd}")
    rr = random.Random(777)
    print("  examples (greedy):")
    for f in SECRETS_C[:3]:
        prefs = torch.tensor([prefix(f, rr)], device=device)
        with torch.no_grad(): e = generer(m1, prefs, greedy=True)
        print(f"    {f} → « {' '.join(e[0])} »")
    print(f"\n  Seal d9c1ebc576 · {time.time()-t0:.0f} s")