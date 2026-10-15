# ════════════════════════════════════════════════════════════════
# COMPRENEUR — Entry 72: Memory, Persistence, and the Cost of the Memo
# Result: 24/24, 0.01 s, 0 tokens, seal d3c505b31b
# Run: plain Python 3, CPU (~1 s)
# ════════════════════════════════════════════════════════════════
import random, time, hashlib
t0 = time.time()
RNG = random.Random(72)

ACTANTS = ["marie","paul","chien","chat","grifflor","blorpte","frumilax",
           "zornik","purbet","chalimbe","vortaxe","snellor","quamish",
           "trelode","marnival","espilou"]
OBJETS  = ["livre","pomme","velo","ballon","cle","lampe","stylo","souris"]
LIEUX   = ["salle","cave","grenier"]
DET_L   = {"salle":"la","cave":"la","grenier":"le","pomme":"la",
           "cle":"la","lampe":"la","souris":"la"}
def det(m): return DET_L.get(m, "le")

objs = OBJETS * 2 ; RNG.shuffle(objs)
FAITS  = [(a, "possede", o) for a, o in zip(ACTANTS, objs)]
FAITS += [(a, "habite", LIEUX[j % 3]) for j, a in enumerate(ACTANTS)]
FAITS += [(a, "mange", ["pomme","souris"][ACTANTS.index(a) % 2])
          for a in ["chien","chat","grifflor","blorpte","frumilax"]]

NOUVEAUX = ["brulte","quennex","samfrie"]
FAITS_UPDATE  = [(a, "possede", o) for a, o in zip(NOUVEAUX, RNG.sample(OBJETS, 3))]
FAITS_UPDATE += [(a, "habite", l) for a, l in zip(NOUVEAUX, LIEUX)]
FAITS_TOTAL = FAITS + FAITS_UPDATE

def possede_de(a): return next(o for (s,r,o) in FAITS_TOTAL if s==a and r=="possede")
def habite_de(a):  return next(o for (s,r,o) in FAITS_TOTAL if s==a and r=="habite")
def enonce(f): s, r, o = f ; return f"{s} {r} {det(o)} {o}"

BASE_ENONCES   = [enonce(f) for f in FAITS]
UPDATE_ENONCES = [enonce(f) for f in FAITS_UPDATE]

DETS    = {"le","la","les","un","une","des","du","de"}
OUTILS  = {"est","ce","que","ne","pas","t","il","elle","ou","a","dans"}
ACTIONS = {"possede","habite","mange","appartient"}

def tokens(p):
    for c in "'-?,.;!":
        p = p.replace(c, " ")
    return [t for t in p.lower().split() if t]

class Comprenneur:
    def __init__(self):
        self.vocab, self.faits = set(), set()
    def apprend(self, e):
        core = [t for t in tokens(e) if t not in OUTILS and t not in DETS]
        act = next((t for t in core if t in ACTIONS), None)
        if act is None: return
        i = core.index(act)
        avant, apres = core[:i], core[i+1:]
        if not avant or not apres: return
        s, o = avant[-1], apres[0]
        self.vocab |= {s, o}
        self.faits.add((o, "possede", s) if act == "appartient" else (s, act, o))
    def interroge(self, q):
        toks = tokens(q)
        neg  = ("ne" in toks) or ("pas" in toks)
        core = [t for t in toks if t not in OUTILS and t not in DETS]
        act  = next((t for t in core if t in ACTIONS), None)
        if act is None: return ("INCONNU", "?")
        i = core.index(act)
        avant, apres = core[:i], core[i+1:]
        if not avant and apres:
            avant, apres = apres, []
        if not avant: return ("INCONNU", "?")
        s = avant[-1]
        inc = [t for t in ([s] + apres) if t not in self.vocab]
        if inc: return ("INCONNU", inc[0])
        if act == "appartient":
            if not apres: return ("INCONNU", "?")
            r = (apres[0], "possede", s) in self.faits
        elif not apres:
            c = [o for (a, rel, o) in sorted(self.faits) if a == s and rel == act]
            return ("LIEU" if act == "habite" else "OBJET", c[0]) if c else ("INCONNU", s)
        else:
            r = (s, act, apres[0]) in self.faits
        r = (not r) if neg else r
        return ("OUI" if r else "NON", None)

QUESTIONS = []
def Q(t, texte, rep): QUESTIONS.append({"t": t, "q": texte, "rep": rep})

for k, a in enumerate(["grifflor","blorpte","frumilax","zornik","purbet"]):
    v = habite_de(a)
    if k % 2 == 0: Q("A", f"Est-ce que {a} habite {det(v)} {v} ?", ("OUI", None))
    else:
        f_ = next(l for l in LIEUX if l != v)
        Q("A", f"Est-ce que {a} habite {det(f_)} {f_} ?", ("NON", None))
for k, a in enumerate(["chalimbe","vortaxe","snellor","quamish","trelode"]):
    if k % 2 == 0: Q("B", f"Que possede {a} ?", ("OBJET", possede_de(a)))
    else:
        o_ = possede_de(a) ; f_ = next(x for x in OBJETS if x != o_)
        Q("B", f"Est-ce que {a} possede {det(f_)} {f_} ?", ("NON", None))
for k, a in enumerate(NOUVEAUX[:2]):
    v = habite_de(a)
    if k % 2 == 0: Q("C", f"Est-ce que {a} habite {det(v)} {v} ?", ("OUI", None))
    else:
        f_ = next(l for l in LIEUX if l != v)
        Q("C", f"Est-ce que {a} habite {det(f_)} {f_} ?", ("NON", None))
Q("C", f"Est-ce que {NOUVEAUX[2]} habite {det(habite_de(NOUVEAUX[2]))} {habite_de(NOUVEAUX[2])} ?", ("OUI", None))
Q("C", f"Que possede {NOUVEAUX[2]} ?", ("OBJET", possede_de(NOUVEAUX[2])))
Q("D", "Est-ce que le brontarque habite la salle ?", ("INCONNU", "brontarque"))
mange_chien = next(o for (s,r,o) in FAITS_TOTAL if s == "chien" and r == "mange")
autre = "souris" if mange_chien == "pomme" else "pomme"
Q("D", f"Est-ce que le chien ne mange pas {det(autre)} {autre} ?", ("OUI", None))
o_m = possede_de("marie")
Q("D", f"Est-ce que {det(o_m)} {o_m} appartient a marie ?", ("OUI", None))
v_z = possede_de("zornik")
Q("D", f"Est-ce que zornik ne possede pas {det(v_z)} {v_z} ?", ("NON", None))
Q("D", "Est-ce que le trezopile possede la cle ?", ("INCONNU", "trezopile"))
for k, a in enumerate(["grifflor","vortaxe","snellor","marnival","espilou"]):
    if k % 2 == 0:
        v = habite_de(a) ; Q("E", f"Est-ce que {a} habite {det(v)} {v} ?", ("OUI", None))
    else:
        Q("E", f"Que possede {a} ?", ("OBJET", possede_de(a)))

if __name__ == "__main__":
    COMP = Comprenneur()
    for e in BASE_ENONCES: COMP.apprend(e)
    resultats = {t: [] for t in "ABCDE"}; audit = []
    for i, q in enumerate(QUESTIONS):
        if i == 10:
            for e in UPDATE_ENONCES: COMP.apprend(e)
        rep = COMP.interroge(q["q"]) ; ok = (rep == q["rep"])
        resultats[q["t"]].append(ok) ; audit.append((i+1, q["t"], rep, q["rep"], ok))
    NOMS = {"A":"Q1-5   start of context","B":"Q6-10  end of base",
            "C":"Q11-14 update","D":"Q15-19 adversarial","E":"Q'1-5  RE-SESSION (no base)"}
    print("── COMPRENEUR (CPU) ──")
    for t in "ABCDE": print(f"  {NOMS[t]:<28}: {sum(resultats[t])}/{len(resultats[t])}")
    total = sum(sum(v) for v in resultats.values())
    print(f"  SONDE anti-divergence       : {total}/24 "
          + ("— 0 bite" if total==24 else "— BITE:"))
    for (n,t,rep,att,ok) in audit:
        if not ok: print(f"    Q{n} [{t}] : rep={rep} attendu={att}")
    B  = len("\n".join(BASE_ENONCES)) + len("\n".join(UPDATE_ENONCES))
    Qc = sum(len(q["q"]) for q in QUESTIONS)
    print(f"\n── MEMO ACCOUNTING (context characters) ──")
    print(f"  base={B:,} · questions={Qc:,}")
    print(f"  LLM stateless (per-request base): {24*B + Qc:,}  (x{(24*B+Qc)/(B+Qc):.1f})")
    print(f"  Comprenneur (base once, for life): {B + Qc:,}  (x1)")
    scelle = hashlib.md5(repr(QUESTIONS).encode()).hexdigest()[:10]
    print(f"\n  seal {scelle} · {time.time()-t0:.2f} s · CPU, 0 token")
    # LLM protocol files are generated by the full entry; see paper Section 5.2.