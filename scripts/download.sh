#!/usr/bin/env bash
# Resumable OSV-5M download (re-run safely; curl -C - continues partial files).
set -u
B=https://huggingface.co/datasets/osv5m/osv5m/resolve/main
mkdir -p data/raw/images/test data/raw/images/train
curl -L -C - -o data/raw/test.csv $B/test.csv &
curl -L -C - -o data/raw/train.csv $B/train.csv &
for i in 00 01 02 03 04; do curl -sL -C - -o data/raw/images/test/$i.zip $B/images/test/$i.zip & done
for i in 00 01; do curl -sL -C - -o data/raw/images/train/$i.zip $B/images/train/$i.zip & done
wait
echo DATA_DONE
