Place downloaded image datasets here (not committed to git).

Expected layout for each source you pass to scripts/build_manifest.py:

  <name>/
    <class-folder>/
      *.png | *.jpg | ...

Class folder names are matched to Gen 1 dex 1-151 using artifacts/label_map.json
(e.g. "Bulbasaur", "001", "bulbasaur", "Mr-Mime").

Example after extracting Kaggle zips:

  data/raw/kvpratama/
    bulbasaur/
    ivysaur/
    ...

Then:

  python scripts/build_manifest.py --data-root data/raw/kvpratama
