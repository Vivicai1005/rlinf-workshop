#!/bin/bash
# Prepare the read-only data volume of the RLinf workshop image: the three models and RoboTwin's assets, at pinned
# revisions, every file checked by sha256. Mount the result at /models/rlinf-workshop in the instance.
#
#   bash prepare_mount.sh <target dir>          # e.g. bash prepare_mount.sh /data/mounts/rlinf-workshop
#
# Result (~33 GB):
#   <target>/lingbot-vla-4b-posttrain-robotwin/   model.safetensors, config.json, lingbotvla_cli.yaml   (16.8 GB)
#   <target>/Qwen2.5-VL-3B-Instruct/              tokenizer, processor and config files                 (11 MB)
#   <target>/lingbot-vla-4b/                      config.json                                           (294 B)
#   <target>/robotwin-assets/                     background_texture/, objects/, embodiments/           (16 GB)
#
# Uses ../docker/fetch.py of this repository. Downloads go through HF_ENDPOINT (default https://hf-mirror.com); large files are fetched as parallel byte ranges.
# Re-running skips files that are already there and correct. Needs python3 and unzip.
set -euo pipefail
T=${1:?usage: prepare_mount.sh <target dir>}
HF=${HF_ENDPOINT:-https://hf-mirror.com}
FETCH="python3 $(dirname "$(readlink -f "$0")")/../docker/fetch.py"
mkdir -p "$T"

ok() { [ -f "$2" ] && [ "$(sha256sum "$2" | cut -d' ' -f1)" = "$1" ]; }
get() {  # get <sha256> <path under target> <url>
    ok "$1" "$T/$2" && { echo "ok      $2"; return; }
    mkdir -p "$(dirname "$T/$2")"
    $FETCH file "$1" "$T/$2" "$3"
}

# Models: Hugging Face repository @ revision, only the files the workshop reads
P=robbyant/lingbot-vla-4b-posttrain-robotwin; PR=3e0c7c476bde3daaac00f79f3741a292a299f60a
Q=Qwen/Qwen2.5-VL-3B-Instruct;               QR=66285546d2b821cf421d4f5eb2576359d3770cd3
L=robbyant/lingbot-vla-4b;                   LR=87712223fb463f0f6a6565f8feba64c35418e393
while read -r sha repo rev name file; do
    get "$sha" "$name/$file" "$HF/$repo/resolve/$rev/$file"
done <<EOF
12153da33b2712d1e17a35a334596f205bb95b6dcd6ad8725ef5d27d6d968752 $P $PR lingbot-vla-4b-posttrain-robotwin model.safetensors
49eb50c482ef102ef39a524556414c7aedb4fb791a443b3b3280049882522437 $P $PR lingbot-vla-4b-posttrain-robotwin config.json
ac7afeb1c71e31d5b5ba92fa600d783ea7eb339783d66d09ed1b12892645fb4c $P $PR lingbot-vla-4b-posttrain-robotwin lingbotvla_cli.yaml
7ed3eed5be6924cc800e8a5e53fc405c1aab1aaf36bad65c33403b36c56827f5 $Q $QR Qwen2.5-VL-3B-Instruct config.json
533f191cc257b7de37a4fccd0a7a1706d75e1aa660f93efaa54e5a2a9f9aace9 $Q $QR Qwen2.5-VL-3B-Instruct generation_config.json
ad60d90252ed0b0705ba14e2d0ad0fec0beac1ea955642b54059b36052d8bc96 $Q $QR Qwen2.5-VL-3B-Instruct chat_template.json
f2058c716eef96ccaed1cc1e2d0c08306b62586d535b28d9d08e691b2fab7ca0 $Q $QR Qwen2.5-VL-3B-Instruct preprocessor_config.json
4abd3520120e266da84c0864fee064d1fb10806f02225911a47253dd38dc5f56 $Q $QR Qwen2.5-VL-3B-Instruct tokenizer_config.json
c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539 $Q $QR Qwen2.5-VL-3B-Instruct tokenizer.json
ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910 $Q $QR Qwen2.5-VL-3B-Instruct vocab.json
599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3 $Q $QR Qwen2.5-VL-3B-Instruct merges.txt
92cebf1aea89c18d30f274eb36f146108413b78f37ec599f9e7e510f4a801c32 $L $LR lingbot-vla-4b config.json
EOF

# RoboTwin assets: dataset TianxingChen/RoboTwin2.0 @ revision, unpacked as RoboTwin's script/_download_assets.sh does
# (the macOS __MACOSX entries are dropped). Nothing else is generated: the image reads them read-only.
A=$T/robotwin-assets; AR=981c92aa34d8f94d4cff47e0d5bc2f7d4e0af042
mkdir -p "$A"
while read -r sha name; do
    if [ -f "$A/$name/.complete" ] && [ "$(cat "$A/$name/.complete")" = "$sha" ]; then echo "ok      robotwin-assets/$name/"; continue; fi
    rm -rf "${A:?}/$name"
    $FETCH file "$sha" "$A/$name.zip" "$HF/datasets/TianxingChen/RoboTwin2.0/resolve/$AR/$name.zip"
    (cd "$A" && unzip -q "$name.zip" -x '__MACOSX/*' && rm "$name.zip")
    echo "$sha" > "$A/$name/.complete"
    echo "unpacked robotwin-assets/$name/"
done <<EOF
54ede0fb5b783e0faa2bc98720d3affd6ca3bb9280b225b48c1aafaf31473070 background_texture
6aa56b3cf1e1064f7c809308144da36b00815f8b137fef2d7e4de856f8becf27 objects
6b87d7d55e106d8ff25917e0538eb1e177fc549280e8a742a8cec3cb9f953fc6 embodiments
EOF

echo
du -sh "$T"/* 2>/dev/null
echo "Done: mount $T read-only at /models/rlinf-workshop"
