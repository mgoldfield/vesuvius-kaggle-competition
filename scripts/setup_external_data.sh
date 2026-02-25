#!/bin/bash
# Setup external scroll data for training
# Fixes nested SCP dirs, symlinks into train dirs, creates combined CSV
set -e

DATA=/workspace/vesuvius-kaggle-competition/data
MIN_FG_PCT=0.5  # Only include volumes with >= 0.5% foreground

echo "=== Setting up external data ==="

# 1. Fix nested dirs from SCP -r
if [ -d "$DATA/external_images/external_images" ]; then
    echo "Flattening nested external_images dir..."
    mv "$DATA/external_images/external_images"/* "$DATA/external_images/"
    rmdir "$DATA/external_images/external_images"
fi
if [ -d "$DATA/external_labels/external_labels" ]; then
    echo "Flattening nested external_labels dir..."
    mv "$DATA/external_labels/external_labels"/* "$DATA/external_labels/"
    rmdir "$DATA/external_labels/external_labels"
fi

# 2. Count files
N_IMG=$(ls "$DATA/external_images/"*.tif 2>/dev/null | wc -l)
N_LBL=$(ls "$DATA/external_labels/"*.tif 2>/dev/null | wc -l)
echo "External images: $N_IMG, labels: $N_LBL"

# 3. Symlink external files into train_images/ and train_labels/
echo "Creating symlinks..."
LINKED=0
for f in "$DATA/external_images/"*.tif; do
    base=$(basename "$f")
    if [ ! -e "$DATA/train_images/$base" ]; then
        ln -s "$f" "$DATA/train_images/$base"
        ((LINKED++))
    fi
done
echo "Linked $LINKED images into train_images/"

LINKED=0
for f in "$DATA/external_labels/"*.tif; do
    base=$(basename "$f")
    if [ ! -e "$DATA/train_labels/$base" ]; then
        ln -s "$f" "$DATA/train_labels/$base"
        ((LINKED++))
    fi
done
echo "Linked $LINKED labels into train_labels/"

# 4. Create combined train CSV (original + filtered external)
echo "Creating combined train CSV (min fg_pct=$MIN_FG_PCT)..."
cp "$DATA/train.csv" "$DATA/train_with_external.csv"

# Append external volumes with sufficient foreground
python3 -c "
import csv
min_fg = $MIN_FG_PCT
added = 0
with open('$DATA/external_volumes.csv') as f_in, \
     open('$DATA/train_with_external.csv', 'a', newline='') as f_out:
    reader = csv.DictReader(f_in)
    writer = csv.writer(f_out)
    for row in reader:
        if float(row['fg_pct']) >= min_fg:
            writer.writerow([row['id'], row['scroll_id']])
            added += 1
print(f'Added {added} external volumes with fg_pct >= {min_fg}%')
"

TOTAL=$(wc -l < "$DATA/train_with_external.csv")
echo "Combined CSV: $((TOTAL - 1)) total volumes"
echo "=== Setup complete ==="
