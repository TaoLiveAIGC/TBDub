# TBDub project page

This branch contains the static site published at <https://taoliveaigc.github.io/TBDub/>.

Videos are stored with the page, encoded for web playback, and requested only after a visitor clicks a poster. The `master` branch contains the model code and its documentation.

V1.1 adds a bilingual update notice and release history (`#v1-1`), plus a GPU
memory comparison and CFG explanation (`#gpu-memory`). Edit `release-copy.js`
for the English and Chinese release text. The original H20 research speed
results remain separate from the new RTX PRO 5000 memory measurements.
The current installation uses one `requirements.txt` with MediaPipe, one BF16
Teacher or Student checkpoint plus shared auxiliary files. Models stay
GPU-resident by default to prioritize speed. If inference runs out of GPU
memory, users can rerun with `--cpu-offload`; the program does not switch
modes automatically.

Preview locally with `python -m http.server 8000 --bind 127.0.0.1` from this
directory. This branch is published directly to GitHub Pages; it does not need
to be merged into the inference branch.
