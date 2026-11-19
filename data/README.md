# data/

The Es Magico sample pack is **confidential** and is not committed (see `.gitignore`).
Download it from the link in the assignment brief and arrange it like this:

```
data/
  labelled/     L01.pdf, L01.json, L02.jpg, L02.json, ... L10.pdf, L10.json
  unlabelled/   U01.pdf, U02.jpg, ... U10.pdf
```

File names are preserved from the pack; a document and its ground truth share the same
stem (`L02.jpg` ↔ `L02.json`). Optional generated sets (also not committed):

```
python scripts/make_stress_set.py --level mild     # -> data/stress_mild/
python scripts/make_stress_set.py --level severe   # -> data/stress_severe/
```
