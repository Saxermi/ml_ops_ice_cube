#!/usr/bin/env bash
set -eo pipefail

# Find every tracked requirements.txt and deduplicate directory names
mapfile -t DIRS < <(
  git ls-files | grep -E '/requirements\.txt$' | xargs -n1 dirname | sort -u
)

echo "Found ${#DIRS[@]} requirement sets:"
printf ' · %s\n' "${DIRS[@]}"

for dir in "${DIRS[@]}"; do
  echo ""
  echo "▶ Updating ${dir}/requirements.txt"
  pushd "${dir}" >/dev/null

  # --ignore-none avoids aborting if a directory has no *.py files (e.g. pure config)
  # --without-referenced makes the file shorter by skipping unused deps; drop if unwanted
  pigar -p requirements.txt -u --ignore-none --without-referenced

  popd >/dev/null
done
