#!/usr/bin/env bash
set -eo pipefail

mapfile -t DIRS < <(
  git ls-files | grep -E '/requirements\.txt$' | xargs -n1 dirname | sort -u
)

echo "Found ${#DIRS[@]} requirement sets:"
printf ' · %s\n' "${DIRS[@]}"

for dir in "${DIRS[@]}"; do
  echo ""
  echo "▶ Updating ${dir}/requirements.txt"
  pushd "${dir}" >/dev/null

  # ▼ New CLI -------------------------------------------------------------
  pigar generate \
        -f requirements.txt \
        --question-answer yes \
        --auto-select \
        --dont-show-differences
  # -----------------------------------------------------------------------
  popd >/dev/null
done

