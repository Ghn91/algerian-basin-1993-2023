"""Writing of tables and figures with their provenance (script, inputs, domain,
commit) recorded in MANIFEST.csv.
"""

import csv
from datetime import datetime
from pathlib import Path

import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    RACINE,
    TAB,
    etiquette_domaine,
    dossier_de_sortie,
    PROVENANCES_EXTERIEURES,
    provenance_declaree,
)


def _chapitre(id_, impose=None):
    return impose or dossier_de_sortie(id_) or "ch04"


MANIFEST = RACINE / "MANIFEST.csv"
COLONNES = [
    "id",
    "type",
    "titre",
    "script",
    "donnees",
    "section_these",
    "domaine",
    "git",
    "date",
]


def _git(racine=None):
    try:
        dot = Path(racine or RACINE) / ".git"
        if dot.is_file():
            cible = dot.read_text(encoding="utf-8").split("gitdir:", 1)[1]
            dot = Path(cible.strip())
        tete = (dot / "HEAD").read_text(encoding="utf-8").strip()
        if not tete.startswith("ref:"):
            return tete[:7]
        ref = tete.split(":", 1)[1].strip()
        fichier = dot / ref
        if fichier.exists():
            return fichier.read_text(encoding="utf-8").strip()[:7]
        empaquetees = dot / "packed-refs"
        if empaquetees.exists():
            for ligne in empaquetees.read_text(encoding="utf-8").splitlines():
                if ligne.endswith(" " + ref):
                    return ligne.split()[0][:7]
        return "non-versionne"
    except Exception:
        return "non-versionne"


def enregistrer(id_, type_, titre, script, donnees, section, domaine=None):
    if domaine is not None and provenance_declaree(domaine) is None:
        raise RuntimeError(
            f"provenance \"{domaine}\" not declared in config.PROVENANCES_EXTERIEURES.\nDeclared: {', '.join(sorted(PROVENANCES_EXTERIEURES)) or 'none'}"
        )
    anciennes = []
    if MANIFEST.exists():
        with open(MANIFEST, encoding="utf-8") as f:
            anciennes = list(csv.DictReader(f))

    for r in anciennes:
        if r["id"] == id_ and r.get("script") and r["script"] != script:
            raise RuntimeError(
                f"identifier {id_} already used by {r['script']} (\"{r.get('titre')}\"); {script} must use another one."
            )

    lignes = [r for r in anciennes if r["id"] != id_]
    lignes.append(
        dict(
            zip(
                COLONNES,
                [
                    id_,
                    type_,
                    titre,
                    script,
                    donnees,
                    section,
                    domaine or etiquette_domaine(),
                    _git(),
                    datetime.now().strftime("%Y-%m-%d %H:%M"),
                ],
            )
        )
    )
    lignes.sort(key=lambda r: r["id"])
    with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLONNES)
        w.writeheader()
        for r in lignes:
            w.writerow({c: r.get(c, "") for c in COLONNES})


def nom_de_fichier(id_, titre):
    return f"{id_}_{titre.lower().replace(' ', '_')}"


def sauver_table(df, id_, titre, script, donnees, section, chapitre=None, domaine=None):
    dossier = TAB / _chapitre(id_, chapitre)
    dossier.mkdir(parents=True, exist_ok=True)
    base = dossier / nom_de_fichier(id_, titre)
    df.to_csv(f"{base}.csv", index=False)
    with open(f"{base}.tex", "w", encoding="utf-8") as f:
        f.write(df.to_latex(index=False, escape=False))
    enregistrer(id_, "tableau", titre, script, donnees, section, domaine)
