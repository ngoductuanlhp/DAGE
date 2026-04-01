

#!/bin/bash

# Set destination directory (modify this path as needed)
DST_DIR="/mnt/localssd/eth3d"

# Create destination directory if it doesn't exist
mkdir -p "$DST_DIR"

echo "Downloading ETH3D scenes to: $DST_DIR"

scenes=("courtyard" "delivery_area" "electro" "facade" "kicker" "meadow" "office" "pipes" "playground" "relief" "relief_2" "terrace" "terrains")
for scene in "${scenes[@]}"; do
    echo "Processing scene: $scene"
    
    # Download to destination directory
    wget -c https://www.eth3d.net/data/${scene}_dslr_depth.7z -P "$DST_DIR"
    
    # Extract to destination directory
    7z x "$DST_DIR/${scene}_dslr_depth.7z" -o"$DST_DIR" -bsp1
    
    # Remove archive
    rm "$DST_DIR/${scene}_dslr_depth.7z"
done

echo "Download and extraction complete!"
