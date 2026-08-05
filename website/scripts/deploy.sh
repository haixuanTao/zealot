#!/usr/bin/env bash
#
# Publish website/dist to the gh-pages branch (GitHub Pages project site at
# https://haixuantao.github.io/zealot/).
#
# Run AFTER a build:  npm run deploy   (= npm run build && this script)
# The wasm demos under public/demos/ are gitignored build output — build them
# with scripts/build-demos.sh first if they changed.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEBSITE_DIR="$(dirname "$SCRIPT_DIR")"
DIST="$WEBSITE_DIR/dist"
BRANCH=gh-pages
REMOTE=${DEPLOY_REMOTE:-origin}

[ -d "$DIST" ] || { echo "no dist/ — run npm run build first" >&2; exit 1; }

# .nojekyll keeps Pages from eating files that start with an underscore.
touch "$DIST/.nojekyll"

# API docs ride along under /doc/: rustdoc for the library crates, with a
# tiny index redirect (there is no root crate to land on). Rebuilt on every
# deploy — it's ~1 s warm and keeps the hosted docs from silently going
# stale relative to the code the site claims to demonstrate.
ZEALOT_DIR="$(dirname "$WEBSITE_DIR")"
(cd "$ZEALOT_DIR" && cargo doc --no-deps -p zealot-env -p zealot-rl >/dev/null)
rm -rf "$DIST/doc"
cp -R "$ZEALOT_DIR/target/doc" "$DIST/doc"
cat > "$DIST/doc/index.html" <<'HTML'
<!DOCTYPE html><meta charset="utf-8">
<meta http-equiv="refresh" content="0; url=zealot_env/index.html">
<a href="zealot_env/index.html">zealot API docs</a>
HTML

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

cd "$WEBSITE_DIR"
SHA=$(git rev-parse --short HEAD)

git init -q "$TMP"
cp -R "$DIST"/. "$TMP"/
cd "$TMP"
git add -A
git -c user.email=deploy@zealot -c user.name=deploy commit -qm "Deploy website — based on $SHA"
git push -q --force "$(cd "$WEBSITE_DIR" && git remote get-url "$REMOTE")" "HEAD:$BRANCH"

echo "Deployed $SHA → $BRANCH (https://haixuantao.github.io/zealot/)"
