name: Validation walk-forward bayésien (bayésien vs fréquentiste vs marché)

# Évalue l'APPORT du modèle bayésien : walk-forward sans fuite temporelle qui
# ajuste DEUX modèles sur le même passé strict à chaque date (fréquentiste +
# bayésien MAP+Laplace) et les compare sur EXACTEMENT les mêmes matchs, avec le
# marché (cotes clôture Pinnacle dévigorisées) comme juge de paix commun.
#
# Répond à : le bayésien bat-il / égale-t-il / fait-il moins bien que le
# fréquentiste sur la PL ? (En L1 : les deux se valaient, l'apport du bayésien
# étant l'incertitude quantifiée, pas un gain de log-loss.)
#
# Pour la validation on n'utilise que la prédiction ponctuelle (au MAP), pas les
# intervalles : pas d'échantillonnage, donc plus rapide.
#
# Écrit data/validation/summary_bayes.csv.
#
# Aucun secret requis. validation/ est à la RACINE, d'où PYTHONPATH=src:.
#
# DURÉE : ce run réajuste deux modèles (dont le bayésien avec hessienne) à chaque
# date d'évaluation. Compter ~20-30 min (le fréquentiste seul prend déjà ~9 min).
# Très loin de la limite runner (6h), mais ce n'est pas instantané.
#
# Cadence : MANUELLE uniquement. On relance quand le modèle ou les données
# changent significativement.

on:
  workflow_dispatch:

permissions:
  contents: write

concurrency:
  group: walkforward-bayes
  cancel-in-progress: false

jobs:
  walkforward-bayes:
    runs-on: ubuntu-latest
    timeout-minutes: 120
    steps:
      - name: Récupérer le repo
        uses: actions/checkout@v4

      - name: Installer Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Installer les dépendances
        run: pip install numpy scipy

      - name: Validation walk-forward bayésien
        run: |
          PYTHONPATH=src:. python -m validation.walkforward_bayes > walkforward_bayes_report.txt 2>&1
          cat walkforward_bayes_report.txt

      - name: Publier le rapport dans le résumé du run
        if: always()
        run: |
          {
            echo "## Validation walk-forward bayésien (bayésien vs fréquentiste vs marché)"
            echo ""
            echo '```'
            cat walkforward_bayes_report.txt 2>/dev/null || echo "(pas de rapport — la validation a échoué avant de produire une sortie)"
            echo '```'
          } >> "$GITHUB_STEP_SUMMARY"

      - name: Committer les résultats de validation
        if: success()
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add data/validation/
          if git diff --staged --quiet; then
            echo "Résultats inchangés — rien à committer."
          else
            git commit -m "validation: walk-forward bayésien $(date -u +%Y-%m-%dT%H:%MZ)"
            git push
          fi

