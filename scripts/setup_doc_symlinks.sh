#!/bin/bash

# Setup documentation symlinks for all repos
# Run this script to recreate symlinks if they get broken

REPOS=(
    "beep-boop"
    "stackr" 
    "satsflow"
    "digi-core"
    "tibocin"
    "bitscrow"
    "lernmi"
    "devao"
)

DIGI_INFRA_PATH="../../digi-infrastructure"
PCS_DOCS_PATH="../../digi-infrastructure/pcs/docs"

for repo in "${REPOS[@]}"; do
    if [ -d "$repo" ]; then
        echo "Setting up symlinks for $repo..."
        
        # Create docs directory if it doesn't exist
        mkdir -p "$repo/docs"
        
        # Create symlinks
        ln -sf "$DIGI_INFRA_PATH/docs" "$repo/docs/digi-infra"
        ln -sf "$PCS_DOCS_PATH" "$repo/docs/pcs"
        
        echo "✅ $repo symlinks created"
    else
        echo "⚠️  $repo directory not found, skipping..."
    fi
done

echo "🎉 Documentation symlinks setup complete!"
echo ""
echo "Each repo now has access to:"
echo "- docs/digi-infra/ -> digi-infrastructure documentation"
echo "- docs/pcs/ -> PCS documentation"
echo ""
echo "These symlinks will always reflect the latest content."
