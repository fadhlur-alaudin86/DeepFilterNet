#!/usr/bin/env bash
# Install (or uninstall) the packaged DeepFilterNet GUI as a Linux desktop app.
#
# Usage:
#   scripts/install-linux.sh [--dist DIR] [--prefix PREFIX] [--icon FILE] [--uninstall]
#
# Defaults: --dist ./dist, --prefix ~/.local. Installs the flet-pack bundle
# binary to <prefix>/bin, writes <prefix>/share/applications/deepfilternet-gui.desktop
# with an absolute Exec path, and refreshes the desktop database. No sudo needed
# for the default prefix. --icon installs a PNG/SVG application icon.
set -euo pipefail

DIST="./dist"
PREFIX="$HOME/.local"
ICON=""
UNINSTALL=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dist) DIST="$2"; shift 2 ;;
        --prefix) PREFIX="$2"; shift 2 ;;
        --icon) ICON="$2"; shift 2 ;;
        --uninstall) UNINSTALL=1; shift ;;
        -h|--help)
            sed -n '2,11p' "$0"
            exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

APP_ID="deepfilternet-gui"
BIN_NAME="DeepFilterNet-GUI"
DESKTOP_FILE="$PREFIX/share/applications/$APP_ID.desktop"

if [[ "$UNINSTALL" -eq 1 ]]; then
    rm -f "$PREFIX/bin/$BIN_NAME" "$DESKTOP_FILE"
    rm -f "$PREFIX/share/icons/hicolor/256x256/apps/$APP_ID.png"
    rm -f "$PREFIX/share/icons/hicolor/scalable/apps/$APP_ID.svg"
    update-desktop-database "$PREFIX/share/applications" 2>/dev/null || true
    echo "Uninstalled $APP_ID."
    exit 0
fi

SRC="$DIST/$BIN_NAME"
if [[ ! -x "$SRC" ]]; then
    echo "error: executable bundle not found at $SRC" >&2
    echo "hint: build it first, e.g. flet pack gui/main.py --name $BIN_NAME ..." >&2
    exit 1
fi

ICON_LINE=""
if [[ -n "$ICON" ]]; then
    if [[ ! -f "$ICON" ]]; then
        echo "error: icon file not found: $ICON" >&2
        exit 1
    fi
    case "$ICON" in
        *.svg|*.SVG)
            ICON_DIR="$PREFIX/share/icons/hicolor/scalable/apps"
            install -Dm644 "$ICON" "$ICON_DIR/$APP_ID.svg"
            ;;
        *)
            ICON_DIR="$PREFIX/share/icons/hicolor/256x256/apps"
            install -Dm644 "$ICON" "$ICON_DIR/$APP_ID.png"
            ;;
    esac
    ICON_LINE="Icon=$APP_ID"
fi

install -Dm755 "$SRC" "$PREFIX/bin/$BIN_NAME"
mkdir -p "$(dirname "$DESKTOP_FILE")"
cat > "$DESKTOP_FILE" <<EOF
[Desktop Entry]
Type=Application
Name=DeepFilterNet GUI
Comment=Speech enhancement with DeepFilterNet
Exec="$PREFIX/bin/$BIN_NAME"
$ICON_LINE
Terminal=false
Categories=AudioVideo;Audio;
StartupWMClass=$BIN_NAME
EOF
# Drop the empty line left behind when no icon is installed.
[[ -z "$ICON_LINE" ]] && sed -i '/^$/d' "$DESKTOP_FILE"

update-desktop-database "$PREFIX/share/applications" 2>/dev/null || true
echo "Installed $APP_ID:"
echo "  binary:  $PREFIX/bin/$BIN_NAME"
echo "  desktop: $DESKTOP_FILE"
