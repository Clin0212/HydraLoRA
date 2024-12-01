#!/bin/bash

cd $(dirname $0)
# Destination path (replace with your defined file path)
CUSTOM_FILE_PATH="modeling_llama.py"

# Get the path to the Transformers library installation
TRANSFORMERS_PATH=$(python3 -c "import transformers; print(transformers.__path__[0])")

# Check if Transformers is installed
if [ -z "$TRANSFORMERS_PATH" ]; then
    echo "Error: Transformers library not found. Please ensure it is installed."
    exit 1
fi

# Determine the path to the target file
TARGET_FILE_PATH="$TRANSFORMERS_PATH/models/llama/modeling_llama.py"

# Check if the target file exists
if [ ! -f "$TARGET_FILE_PATH" ]; then
    echo "Error: Target file not found at $TARGET_FILE_PATH."
    exit 1
fi

# Backup original files
BACKUP_FILE_PATH="${TARGET_FILE_PATH}.bak"
if [ ! -f "$BACKUP_FILE_PATH" ]; then
    echo "Creating a backup of the original file..."
    cp "$TARGET_FILE_PATH" "$BACKUP_FILE_PATH"
fi

# Replacement of files
echo "Replacing $TARGET_FILE_PATH with $CUSTOM_FILE_PATH..."
rm "$TARGET_FILE_PATH"
ln -s $(realpath "$CUSTOM_FILE_PATH") "$TARGET_FILE_PATH"

# Check replacement results
if [ $? -eq 0 ]; then
    echo "Replacement successful! Backup file is located at $BACKUP_FILE_PATH."
else
    echo "Error: Replacement failed."
    exit 1
fi
