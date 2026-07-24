#!/usr/bin/env bash
# =============================================================================
# check_namespace_convention.sh — enforce "namespace = file location"
#   mpmstd::a::b::Name  <=>  defined in src/a/b/
# See docs/namespace_convention.md. Run via `make lint`.
#
# Checks (violations printed; exit 1 if any):
#   D1/D2/D3  each src file's block namespace matches its directory
#             (cpu/ gpu/ stripped; kernels/detail allowed as extra leaf;
#              src/common/ lives at mpmstd:: root; single-line forward decls
#              like `namespace mpmstd::x { class Y; }` are ignored).
#   C2        no cross-module re-export alias (using X = <other module>::...) in headers.
#   C3        no `using namespace` in headers.
#   C4        no <=2-char namespace aliases anywhere.
# The orphan test/ tree is excluded (merged in stage 4).
# =============================================================================
set -u
cd "$(dirname "$0")/.."
TMP="$(mktemp)"
trap 'rm -f "$TMP"' EXIT

# ---- D1/D2/D3: block namespace matches directory ---------------------------
while IFS= read -r f; do
  rel="${f#src/}"; dir="$(dirname "$rel")"
  exp="$(printf '%s' "$dir" | sed -E 's#/(cpu|gpu)$##')"
  case "$exp" in
    common|common/*) expns="mpmstd" ;;
    *)               expns="mpmstd::$(printf '%s' "$exp" | sed 's#/#::#g')" ;;
  esac
  # multi-line block namespaces only: drop lines that also close on themselves
  # (`{ ... }` on one line = forward declaration / tiny inline).
  while IFS= read -r ns; do
    [ -z "$ns" ] && continue
    case "$ns" in
      "$expns"|"$expns::kernels"|"$expns::detail"|"mpmstd"|"mpmstd::common") ;;
      *) echo "[D1] $f: block namespace '$ns' != expected '$expns'" >> "$TMP" ;;
    esac
  done < <(grep -E 'namespace mpmstd(::[A-Za-z_]+)+[[:space:]]*\{' "$f" \
             | grep -vE '\}' \
             | grep -oE 'mpmstd(::[A-Za-z_]+)+' | sort -u)
done < <(find src -type f \( -name '*.hpp' -o -name '*.cpp' -o -name '*.cu' -o -name '*.cuh' \))

# ---- C2: cross-module re-export alias in headers ---------------------------
grep -rnE 'using[[:space:]]+[A-Za-z_]+[[:space:]]*=[[:space:]]*(parallel|grid|boundary|config|equation|physics|linear_solver|driver|post)::' \
  src --include=*.hpp --include=*.cuh 2>/dev/null | sed 's/^/[C2] /' >> "$TMP"

# ---- C3: using namespace in a header ---------------------------------------
grep -rnE '^[[:space:]]*using[[:space:]]+namespace[[:space:]]' \
  src --include=*.hpp --include=*.cuh 2>/dev/null | sed 's/^/[C3] /' >> "$TMP"

# ---- C4: <=2-char namespace alias ------------------------------------------
grep -rnE 'namespace[[:space:]]+[A-Za-z]{1,2}[[:space:]]*=' \
  src apps --include=*.hpp --include=*.cpp --include=*.cu --include=*.cuh 2>/dev/null \
  | sed 's/^/[C4] /' >> "$TMP"

n="$(wc -l < "$TMP" | tr -d ' ')"
if [ "$n" -gt 0 ]; then
  echo "namespace-convention: $n violation(s):"
  cat "$TMP"
  exit 1
fi
echo "namespace-convention: OK"
