#!/bin/bash
# Download all Elevant benchmark JSONL files

# Directory to save files
OUTPUT_DIR="benchmarks_data"
mkdir -p "$OUTPUT_DIR"

# List of URLs to download
URLS=(
"https://raw.githubusercontent.com/ad-freiburg/elevant/master/benchmarks/aida-conll-test.benchmark.jsonl"
"https://raw.githubusercontent.com/ad-freiburg/elevant/master/benchmarks/derczynski.benchmark.jsonl"
"https://raw.githubusercontent.com/ad-freiburg/elevant/master/benchmarks/kore50.benchmark.jsonl"
"https://raw.githubusercontent.com/ad-freiburg/elevant/master/benchmarks/msnbc-updated.benchmark.jsonl"
"https://raw.githubusercontent.com/ad-freiburg/elevant/master/benchmarks/news-fair-no-coref.benchmark.jsonl"
"https://raw.githubusercontent.com/ad-freiburg/elevant/master/benchmarks/news-fair-v2-no-coref.benchmark.jsonl"
"https://raw.githubusercontent.com/ad-freiburg/elevant/master/benchmarks/oke-2015-eval.benchmark.jsonl"
"https://raw.githubusercontent.com/ad-freiburg/elevant/master/benchmarks/reuters-128.benchmark.jsonl"
"https://raw.githubusercontent.com/ad-freiburg/elevant/master/benchmarks/wiki-fair-no-coref.benchmark.jsonl"
"https://raw.githubusercontent.com/ad-freiburg/elevant/master/benchmarks/wiki-fair-v2-no-coref.benchmark.jsonl"
"https://raw.githubusercontent.com/ad-freiburg/elevant/master/benchmarks/spotlight.benchmark.jsonl"
"https://raw.githubusercontent.com/ad-freiburg/elevant/master/benchmarks/rss-500.benchmark.jsonl"
)

echo "Downloading ${#URLS[@]} benchmark files to '$OUTPUT_DIR'..."

if command -v wget >/dev/null 2>&1; then
    for url in "${URLS[@]}"; do
        echo "Downloading $(basename "$url")"
        wget -q -P "$OUTPUT_DIR" "$url"
    done
elif command -v curl >/dev/null 2>&1; then
    for url in "${URLS[@]}"; do
        echo "Downloading $(basename "$url")"
        curl -s -L -o "$OUTPUT_DIR/$(basename "$url")" "$url"
    done
else
    echo "Error: need wget or curl to download files."
    exit 1
fi

echo "Done! All files are saved in: $OUTPUT_DIR"
