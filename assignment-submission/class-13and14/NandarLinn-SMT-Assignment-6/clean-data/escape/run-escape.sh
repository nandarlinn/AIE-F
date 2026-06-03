#!/bin/bash
# Define the path to the Moses tokenizer scripts
SCRIPTS_DIR="/home/elio/tool/mosesbin/ubuntu-17.04/moses/scripts/tokenizer"
ESCAPE_SCRIPT="$SCRIPTS_DIR/escape-special-chars.perl"
# Check if the script exists before running
if [ ! -f "$ESCAPE_SCRIPT" ]; then
echo "Error: escape-special-chars.perl not found at $ESCAPE_SCRIPT"
exit 1
fi
echo "Starting character escaping…"
# Loop through all files ending in .my and .th
for file in *.my *.ph; do
# Skip if the file doesn't exist (in case no files match)
[ -e "$file" ] || continue
echo "Processing: $file"
# Run the escaping script and output to <filename>.escape
cat "$file" | perl "$ESCAPE_SCRIPT" > "$file.escape"
done
echo "Done! Escaped files have been created with the .escape suffix."