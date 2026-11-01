"""Show how each document is split into legally meaningful units — no models needed."""
from collections import Counter

from _common import label_of, load_corpus

for name, doc in load_corpus().items():
    chunks = doc.all_chunks()
    kinds = Counter(f"{c.chunk_type.value}/{c.extraction_confidence.value}" for c in chunks)
    leaves = sum(len(c.leaf_units()) for c in chunks)
    print(f"\n== {doc.title}  [{doc.jurisdiction}, {doc.document_type.value}]")
    print(f"   {len(chunks)} chunks → {leaves} indexable leaf units, {len(doc.sections)} sections")
    print(f"   {dict(kinds)}")
    for c in chunks[:4]:
        split = f"  (split into {len(c.sub_chunks)} sub-chunks)" if not c.is_atomic else ""
        print(f"   [{c.order:>3}] {c.chunk_type.value:<24} {label_of(c, 58)}{split}")
    if len(chunks) > 4:
        print("   ...")
