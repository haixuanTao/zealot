#!/bin/bash
# Build everything (wasm demo + site). Deploy the resulting `build/` directory
# to any static host — no server-side code is needed (but the host must serve
# .wasm with the application/wasm MIME type; the bundled .htaccess handles
# Apache hosts).
#
#   rsync -av --delete-after build/ user@host:/path/to/site

set -e
npm run build:all
cp .htaccess build/.
echo "Site built in build/ — rsync it to your static host."
