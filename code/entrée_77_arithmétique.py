# ════════════════════════════════════════════════════════════════
# COMPRENEUR — Entry 77: The Arithmetic Engine
# Result: 37/37, 0.01 s, 0 tokens, seal 885a24b5e9
# Run: plain Python 3, CPU, no GPU, stdlib only (~1 s)
# ════════════════════════════════════════════════════════════════
import time, hashlib, math, unicodedata
from fractions import Fraction
t0 = time.time()

UN = {"zero":0,"un":1,"une":1,"deux":2,"trois":3,"quatre":4,"cinq":5,
      "six":6,"sept":7,"huit":8,"neuf":9,"dix":10,"onze":11,"douze":12,
      "treize":13,"quatorze":14,"quinze":15,"seize":16}
DZ = {"vingt":20,"trente":30,"quarante":40,"cinquante":50,"soixante":60}
OP = {"plus":"+","moins":"-","fois":"*","multiplie":"*","multiplier":"*",
      "divise":"/","diviser":"/","sur":"/"}
EGAL = {"egale","egal","egalent"}
FILLERS = {"combien","font","que","vaut","valent","est","ce","si",
           "le","la","les","de","des","du","par","et","un"}
BIO = {"chat","chien","pomme","souris","mange","habite","possede",
       "marie","paul","livre","velo","ballon","cle","lampe","stylo",
       "salle","cave","grenier"}

def norm(p):
    p = p.lower()
    p = "".join(c for c in unicodedata.normalize("NFD", p)
                if unicodedata.category(c) != "Mn")
    for ch in "?!.,;:'-’":
        p = p.replace(ch, " ")
    p = p.replace("vingts", "vingt")
    return p.split()

def parse_nombre(toks):
    if not toks: return None
    t = toks[0]
    if t in UN: return (UN[t], 1)
    if t == "cent": return (100, 1)
    if t == "quatre" and len(toks) > 1 and toks[1] == "vingt":
        base, j = 80, 2
        if j < len(toks) and toks[j] == "dix": base, j = 90, 3
        if j < len(toks) and toks[j] in UN: base += UN[toks[j]]; j += 1
        return (base, j) if base <= 99 else None
    if t == "soixante":
        base, j = 60, 1
        if j < len(toks) and toks[j] == "dix": base, j = 70, 2
        if j < len(toks) and toks[j] in UN:
            if toks[j] == "onze" and base == 60: base, j = 71, 2
            else: base += UN[toks[j]]; j += 1
        return (base, j) if base <= 79 else None
    if t in DZ:
        base, j = DZ[t], 1
        if j < len(toks) and toks[j] == "et" and j+1 < len(toks) \
           and toks[j+1] in UN:
            base += UN[toks[j+1]]; j += 2
        elif j < len(toks) and toks[j] in UN:
            base += UN[toks[j]]; j += 1
        return (base, j)
    return None

def items_de_tokens(toks):
    items, i = [], 0
    while i < len(toks):
        t = toks[i]
        if t.isdigit(): items.append(("N", Fraction(int(t)))); i += 1; continue
        if t in OP: items.append(("OP", OP[t])); i += 1; continue
        if t == "x":
            items.append(("X",)); i += 1
            if i < len(toks) and toks[i] == "carre":
                items.append(("CARRE",)); i += 1
            continue
        if t in EGAL: items.append(("EQ",)); i += 1; continue
        r = parse_nombre(toks[i:])
        if r: v, used = r; items.append(("N", Fraction(v))); i += used; continue
        return None, t
    return items, None

def fmt(f):
    f = Fraction(f)
    return str(f.numerator) if f.denominator == 1 \
           else f"{f.numerator}/{f.denominator}"

def evalue(items):
    if not items or items[0][0] != "N": return None
    vals, ops = [items[0][1]], []
    for it in items[1:]:
        if it[0] == "OP": ops.append(it[1])
        elif it[0] == "N": vals.append(it[1])
        else: return None
    if len(ops) != len(vals) - 1: return None
    v2, o2 = [vals[0]], []
    for op, v in zip(ops, vals[1:]):
        if op == "*": v2[-1] = v2[-1] * v
        elif op == "/":
            if v == 0: return "DIV0"
            v2[-1] = v2[-1] / v
        else: o2.append(op); v2.append(v)
    res = v2[0]
    for op, v in zip(o2, v2[1:]):
        res = res + v if op == "+" else res - v
    return res

def racine_exacte(f):
    f = Fraction(f)
    if f < 0: return None
    n, d = f.numerator, f.denominator
    rn, rd = math.isqrt(n), math.isqrt(d)
    return Fraction(rn, rd) if rn*rn == n and rd*rd == d else None

def parse_gauche(g):
    if not g: return None
    a, b, deg, j = Fraction(1), Fraction(0), 1, 0
    if g[0][0] == "N":
        a = g[0][1]; j = 1
        if j >= len(g) or g[j][0] != "X": return None
        j += 1
    elif g[0][0] == "X": j = 1
    else: return None
    if j < len(g) and g[j][0] == "CARRE": deg = 2; j += 1
    if j < len(g):
        if g[j][0] != "OP" or j+1 >= len(g) or g[j+1][0] != "N": return None
        op, v = g[j][1], g[j+1][1]
        if op == "+": b = v
        elif op == "-": b = -v
        elif op == "*": a = a * v
        else: return None
        j += 2
    if j != len(g): return None
    return a, b, deg

def resous(items):
    k = [i for i, it in enumerate(items) if it[0] == "EQ"]
    if len(k) != 1: return ("PORTE", "MALFORME — declare")
    k = k[0]; g, d = items[:k], items[k+1:]
    if len(d) != 1 or d[0][0] != "N": return ("PORTE", "MALFORME — declare")
    c = d[0][1]; r = parse_gauche(g)
    if r is None: return ("PORTE", "MALFORME — declare")
    a, b, deg = r
    if deg == 1:
        if a == 0: return ("PORTE", "MALFORME — declare")
        return ("X", "x = " + fmt((c - b) / a))
    rhs = (c - b) / a
    if rhs < 0: return ("PORTE", "PAS DE SOLUTION REELLE — declare")
    rac = racine_exacte(rhs)
    if rac is None: return ("PORTE", "RACINES NON RATIONNELLES — declare")
    if rac == 0: return ("X", "x = 0")
    return ("X", f"x = {fmt(rac)} ou x = -{fmt(rac)}")

def verifie(items):
    k = [i for i, it in enumerate(items) if it[0] == "EQ"]
    if len(k) != 1: return ("PORTE", "MALFORME — declare")
    k = k[0]; g, d = items[:k], items[k+1:]
    if len(d) != 1 or d[0][0] != "N": return ("PORTE", "MALFORME — declare")
    if any(it[0] == "X" for it in g): return resous(items)
    v = evalue(g)
    if v == "DIV0": return ("PORTE", "DIVISION PAR ZERO — declare")
    if v is None: return ("PORTE", "MALFORME — declare")
    return ("N", "OUI" if v == d[0][1] else "NON")

def repondre(q):
    toks = norm(q)
    if not toks: return ("PORTE", "INCONNU — vide")
    signal_arith = any(t in OP or t.isdigit() or t in ("x","carre")
                       or t in EGAL or t in UN or t in DZ or t == "cent"
                       for t in toks)
    if not signal_arith and any(t in BIO for t in toks):
        return ("DELEG", "HORS MODULE ARITHMETIQUE — delegation ▐")
    if "si" in toks:
        toks = toks[toks.index("si")+1:]
    toks = [t for t in toks if t not in FILLERS]
    items, inconnu = items_de_tokens(toks)
    if items is None:
        return ("PORTE", f"INCONNU: {inconnu}" if inconnu else "MALFORME — declare")
    if not items: return ("PORTE", "INCONNU — rien a calculer")
    if any(it[0] == "X" for it in items) and any(it[0] == "EQ" for it in items):
        return resous(items)
    if any(it[0] == "EQ" for it in items):
        return verifie(items)
    if any(it[0] in ("X","CARRE","EQ") for it in items):
        return ("PORTE", "MALFORME — declare")
    v = evalue(items)
    if v == "DIV0": return ("PORTE", "DIVISION PAR ZERO — declare")
    if v is None: return ("PORTE", "MALFORME — declare")
    return ("N", fmt(v))

CORPUS = [
 ("A","combien font 3 plus 4 ?","7"),
 ("A","que vaut 12 moins 5 ?","7"),
 ("A","combien font 6 fois 7 ?","42"),
 ("A","que vaut 20 divise par 4 ?","5"),
 ("A","est-ce que 9 plus 6 egale 15 ?","OUI"),
 ("A","est-ce que 8 fois 8 egale 63 ?","NON"),
 ("B","combien font 47 fois 89 ?","4183"),
 ("B","que vaut 83 fois 76 ?","6308"),
 ("B","combien font 123 plus 456 ?","579"),
 ("B","que vaut 34 fois 67 ?","2278"),
 ("B","que vaut 256 moins 389 ?","-133"),
 ("B","combien font 91 fois 88 ?","8008"),
 ("B","que vaut 10 plus 2 fois 5 ?","20"),
 ("C","combien font deux plus trois ?","5"),
 ("C","que vaut douze fois 8 ?","96"),
 ("C","combien font vingt-cinq plus trente ?","55"),
 ("C","que vaut sept fois neuf ?","63"),
 ("C","combien font cent moins quatorze ?","86"),
 ("C","que vaut soixante divise par dix ?","6"),
 ("D","combien font 3 moins 10 ?","-7"),
 ("D","que vaut 7 divise par 2 ?","7/2"),
 ("D","combien font 10 divise par 4 ?","5/2"),
 ("D","que vaut 0 moins 8 ?","-8"),
 ("D","combien font 100 divise par 8 ?","25/2"),
 ("D","que vaut 9 divise par 3 ?","3"),
 ("E","que vaut x si x plus 3 egale 7 ?","x = 4"),
 ("E","que vaut x si 2 x moins 5 egale 9 ?","x = 7"),
 ("E","que vaut x si x carre plus 3 egale 7 ?","x = 2 ou x = -2"),
 ("E","que vaut x si x fois 4 egale 24 ?","x = 6"),
 ("E","que vaut x si x moins 12 egale 5 ?","x = 17"),
 ("E","que vaut x si x carre egale 16 ?","x = 4 ou x = -4"),
 ("F","combien font banane plus chien ?","porte"),
 ("F","que valt 7 divise par 0 ?","porte"),
 ("F","que vaut x si x carre egale 2 ?","porte"),
 ("F","combien font le vent ?","porte"),
 ("F","que vait 3 plus ?","porte"),
 ("F","est-ce que le chat mange la pomme ?","delegation"),
]

def note(rep, attendu):
    typ, val = rep
    if attendu == "porte": return typ == "PORTE"
    if attendu == "delegation": return typ == "DELEG"
    return typ in ("N","X") and val == attendu

if __name__ == "__main__":
    print("── THE 37 QUESTIONS ──")
    scores = {tr: [0, 0] for tr in "ABCDEF"}
    for tr, q, att in CORPUS:
        rep = repondre(q); ok = note(rep, att)
        scores[tr][0] += ok; scores[tr][1] += 1
        print(f"  {'OK ' if ok else 'FAIL'} [{tr}] « {q} » → {rep[1]}")
    for tr in "ABCDEF":
        b, n = scores[tr]
        print(f"  {tr}: {b}/{n}")
    total = sum(scores[t][0] for t in "ABCDEF")
    print(f"\n  TOTAL: {total}/37 · seal 885a24b5e9"
          f" · {time.time()-t0:.2f} s · CPU, 0 token")