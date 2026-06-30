#!/bin/sh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" 2>/dev/null && pwd)"

if [ ! -f "$SCRIPT_DIR/vec" ] || [ ! -f "$SCRIPT_DIR/vec-cpu" ] || [ ! -f "$SCRIPT_DIR/test" ]; then
    echo "Binaries not found in $SCRIPT_DIR. Build first with:  ./build.sh"
    exit 1
fi

echo "Installing VEC to /usr/local/bin/..."

cp "$SCRIPT_DIR/vec"       /usr/local/bin/vec
cp "$SCRIPT_DIR/vec-cpu"   /usr/local/bin/vec-cpu
cp "$SCRIPT_DIR/test"      /usr/local/bin/vec-test

chmod 755 /usr/local/bin/vec /usr/local/bin/vec-cpu /usr/local/bin/vec-test

echo "Done."
echo ""
echo "=== VEC installed ==="
echo ""
echo "Start a named database from any folder:"
echo "  vec name 1024"
echo ""
echo "Start the CPU-only build:"
echo "  vec-cpu name 1024"
echo ""
echo "Run the integration test:"
echo "  vec-test"
echo ""
echo "Stopped instance auto-saves. All files land in the current directory."
echo ""
echo "Created by PsyChip (root@psychip.net)"
