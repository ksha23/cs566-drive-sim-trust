"""Extract Para-Lane scenes from a subset of the 256 MiB byte-range parts of the split ZIP, by writing the parts we
have into a sparse file of the full size and letting zipfile read only the members we need (SHA-256 checked)."""
import glob, hashlib, os, sys, zipfile
from common.config import DATA

P, T = 268435456, 89712923963  # part size, full archive size
d = f"{DATA}/paralane"; sp = f"{d}/sparse.zip"
sums = dict(l.split()[::-1] for l in open(f"{d}/parts/SHA256SUMS") if l.strip())
with open(sp, "wb") as f:
    f.truncate(T)
    for p in sorted(glob.glob(f"{d}/parts/part-*")):
        i = int(p.split("-")[-1]); b = open(p, "rb").read()
        ok = hashlib.sha256(b).hexdigest() == sums.get(f"paralane_v_1_0.zip.part-{i:05d}")
        print(p, "sha ok" if ok else "SHA MISMATCH", flush=True); f.seek(i * P); f.write(b)
z = zipfile.ZipFile(sp)
for s in sys.argv[1:]:
    n = 0
    for e in z.infolist():
        if e.filename.startswith(f"0315/{s}/"):
            z.extract(e, f"{d}/extracted"); n += 1
    print(s, "extracted", n, "files", flush=True)
os.remove(sp)
