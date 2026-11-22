#!/usr/bin/env bash
# Garbage-collect expired GitHub-side data (OPM SD7 · Garbage-collecting GitHub).
#   GH_REPO=owner/repo GH_TOKEN=... scripts/gh_cleanup.sh [--dry-run]
# Retention days mirror config.Retention (gh_cache_days / gh_artifact_days / gh_run_days);
# tests/test_cicd.py fails if the defaults below drift from config.py.
set -euo pipefail
GH_CACHE_DAYS="${GH_CACHE_DAYS:-3}"
GH_ARTIFACT_DAYS="${GH_ARTIFACT_DAYS:-30}"
GH_RUN_DAYS="${GH_RUN_DAYS:-30}"
DRY=""; [[ "${1:-}" == "--dry-run" ]] && DRY="echo [dry-run]"
: "${GH_REPO:?set GH_REPO=owner/repo}"
cutoff() { date -u -d "-$1 days" +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || date -u -v-"$1"d +%Y-%m-%dT%H:%M:%SZ; }

cache_prefixes=(driftrev-data- driftrev-de-data- driftrev-site-us- driftrev-site-de-)
echo "== Actions caches: keep the newest per prefix (${cache_prefixes[*]}), delete others older than ${GH_CACHE_DAYS}d"
newest_ids="0"
for prefix in "${cache_prefixes[@]}"; do
  newest=$(gh cache list --repo "$GH_REPO" --key "$prefix" --sort created_at --order desc --limit 1 --json id --jq '.[0].id // empty')
  newest_ids="$newest_ids,${newest:-0}"
done
c=$(cutoff "$GH_CACHE_DAYS")
gh cache list --repo "$GH_REPO" --limit 100 --json id,key,createdAt \
  --jq ".[] | select(.createdAt < \"$c\") | select(.id as \$id | [$newest_ids] | index(\$id) | not) | .id" |
  while read -r id; do $DRY gh cache delete "$id" --repo "$GH_REPO"; done

echo "== Workflow artifacts older than ${GH_ARTIFACT_DAYS}d"
c=$(cutoff "$GH_ARTIFACT_DAYS")
gh api --paginate "repos/$GH_REPO/actions/artifacts?per_page=100" \
  --jq ".artifacts[] | select(.created_at < \"$c\") | select(.expired | not) | .id" |
  while read -r id; do $DRY gh api -X DELETE "repos/$GH_REPO/actions/artifacts/$id"; done

echo "== Workflow runs older than ${GH_RUN_DAYS}d"
c=$(cutoff "$GH_RUN_DAYS")
gh run list --repo "$GH_REPO" --limit 500 --json databaseId,createdAt,status \
  --jq ".[] | select(.createdAt < \"$c\") | select(.status == \"completed\") | .databaseId" |
  while read -r id; do $DRY gh run delete "$id" --repo "$GH_REPO" || true; done

echo "== Snapshot release: keep only the newest asset"
if gh release view data-snapshot --repo "$GH_REPO" >/dev/null 2>&1; then
  gh release view data-snapshot --repo "$GH_REPO" --json assets \
    --jq '.assets | sort_by(.createdAt) | reverse | .[1:] | .[].name' |
    while read -r name; do $DRY gh release delete-asset data-snapshot "$name" --repo "$GH_REPO" -y; done
fi
echo "cleanup done"
