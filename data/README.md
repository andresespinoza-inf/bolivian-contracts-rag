# Data provenance

| File | What it is | Real personal data? |
|---|---|---|
| `synthetic/contrato_bo_sintetico.txt` | Bolivian purchase contract of a plot of land (sale of an undivided share). Derived from the author's own contract; **every name, ID number, registry number and place name was replaced with fictional values**. The signature block deliberately names different people than clause one. Text keeps PDF-extraction quirks (wrapped headings). | No |
| `synthetic/contrato_es_sintetico.txt` | Spanish new-build purchase contract. Original text written for this project, modelled on the *structure* of public notarial templates (recitals, compound ordinal clauses, a long payment clause). No third-party text is reproduced. | No |
| `legal/codigo_civil_bo_extracto.txt` | Hand-compiled extract of the Bolivian Civil Code (sale; contract formation; consent defects; nullity and annulability; registry publicity). Headings starting with `#` were added by the author to group topics. Public legal text — **verify against the official source before relying on it**; it is not complete. | No |

Real client documents must never be added to this repository (`data/private/` and `*.pdf` are git-ignored).
