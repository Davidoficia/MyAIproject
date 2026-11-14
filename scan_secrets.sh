#!/usr/bin/env bash
# Varre os arquivos que vão pro commit atrás de padrões de token/chave.
# Uso: ./scan_secrets.sh  (falha com código 1 se achar algo)
set -u
PADROES='ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|gho_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----|hf_[A-Za-z0-9]{20,}'
ALVOS=$(git diff --cached --name-only 2>/dev/null; git ls-files 2>/dev/null)
achou=0
for f in $(echo "$ALVOS" | sort -u); do
  [ -f "$f" ] || continue
  if grep -EnH "$PADROES" "$f" >/dev/null 2>&1; then
    echo "ACHEI possível segredo em: $f"
    achou=1
  fi
done
if [ "$achou" -eq 1 ]; then echo "Bloqueado. Remova antes de commitar."; exit 1; fi
echo "Varredura limpa."
