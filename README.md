# 👥 Prédiction de l'attrition des employés

**Qui risque de partir, et pourquoi ?** Comparaison de trois méthodes de classification — régression logistique, LDA et QDA — sur 1 001 employés, avec un choix final arbitré sur le coût métier plutôt que sur le score.

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![AUC](https://img.shields.io/badge/AUC-0.891-success)
![Rappel](https://img.shields.io/badge/Rappel-84%25-success)

---

## Le problème

Un départ non anticipé coûte cher : recrutement, formation, perte de savoir-faire. Une fausse alerte coûte un entretien.

Cette asymétrie structure tout le projet. Sur 1 001 employés, **17,3 % sont partis** — des classes déséquilibrées qui rendent l'exactitude trompeuse : un modèle qui ne détecterait jamais un départ afficherait encore 82,7 % de bonnes réponses, sans aucune utilité.

---

## Le résultat le plus intéressant : quand le modèle le plus complexe s'auto-annule

Trois méthodes sont comparées, représentant deux philosophies opposées.

**L'approche discriminative** — la régression logistique modélise directement P(Y|X). Elle trace une frontière sans hypothèse sur la distribution des données.

**L'approche générative** — LDA et QDA modélisent P(X|Y) par classe via une loi normale, puis appliquent le théorème de Bayes. La LDA suppose une covariance commune aux deux classes et produit une frontière droite ; la QDA estime une covariance par classe et autorise une frontière courbe, donc plus flexible.

La QDA est *a priori* la plus puissante des trois. Sauf que la validation croisée retient pour elle un **paramètre de régularisation de 1,0** — c'est-à-dire la valeur qui ramène sa covariance à une covariance partagée, donc à un comportement de LDA.

**Les données elles-mêmes rejettent la flexibilité supplémentaire.** La QDA optimale *est* une LDA. Ce n'est pas un échec du modèle, c'est une information : l'hypothèse de covariance commune est la bonne ici, et une complexité additionnelle non justifiée par les données ne produit aucun gain.

---

## Trois modèles, un choix qui ne se joue pas sur l'AUC

| Modèle | AUC test | Rappel (départs) | Précision (départs) | Interprétabilité |
|---|---:|---:|---:|---|
| **Régression logistique** | **0,891** | **0,84** | 0,38 | Odds ratios, p-values |
| LDA | 0,891 | 0,33 | 0,88 | Coefficients, sans p-values |
| QDA | 0,884 | 0,21 | 0,90 | Aucune |

Les courbes ROC sont **quasiment superposées** : le pouvoir de classement global est identique. Choisir sur l'AUC seule n'aurait aucun sens ici.

La différence se joue entièrement sur le comportement au seuil de 0,5. La logistique **détecte 84 % des départs réels** au prix de nombreuses fausses alertes. La LDA et la QDA sont beaucoup plus précises quand elles annoncent un départ — mais elles en ratent respectivement 67 % et 79 %.

**La régression logistique est retenue.** Dans un contexte RH où un départ manqué coûte bien plus qu'un entretien inutile, privilégier le rappel est l'arbitrage cohérent. S'y ajoute un second avantage décisif : elle est la seule à fournir des odds ratios et des p-values, donc à pouvoir **expliquer sa décision aux équipes RH** — un modèle que personne ne comprend ne sera pas utilisé.

Comme le formule la présentation du projet : *« plus complexe » n'est pas toujours « meilleur » — la flexibilité d'une méthode doit être justifiée par les données, pas supposée a priori.*

---

## Ce qui fait partir un employé

Les coefficients standardisés sont convertis en **odds ratios**, lisibles pour une augmentation d'un écart-type.

| Variable | Odds ratio | Effet |
|---|---:|---|
| Heures travaillées / semaine | **1,47** | 🔴 Risque |
| Distance domicile-travail | **1,37** | 🔴 Risque |
| Satisfaction au travail | ~0,70 | 🟢 Protection |
| Ancienneté | ~0,70 | 🟢 Protection |
| Salaire mensuel | ~0,70 | 🟢 Protection |
| Département, télétravail | ≈ 1,00 | ⚪ Sans effet |

Les profils moyens confirment ces écarts de façon très concrète. Un employé qui part parcourt **11,94 km** contre 7,15 km pour celui qui reste, travaille **46,3 heures** contre 42,5, déclare une satisfaction de **2,73** contre 3,34, compte **2,53 ans** d'ancienneté contre 4,41, et gagne nettement moins.

**Validation croisée des méthodes.** Les coefficients de la LDA suivent exactement la même hiérarchie que les odds ratios de la logistique. Cette concordance entre deux approches de nature différente — l'une discriminative, l'autre générative — renforce considérablement la confiance dans les variables identifiées.

Résultat secondaire utile : le **département et le télétravail n'ont aucun pouvoir prédictif** (odds ratios ≈ 1). Segmenter la politique de rétention par service serait une fausse piste.

---

## Rigueur méthodologique

### Un nettoyage fondé sur l'observation, pas sur des suppositions

Le diagnostic du fichier brut révèle quatre types d'anomalies, chacune traitée séparément.

Des **âges hors bornes métier** (au-delà de 18-70 ans pour des employés actifs). Des **valeurs sentinelles** sur la satisfaction : la distribution réelle est continue sur [0, 5], et les valeurs isolées à −1,0 et 8,0 sont des codes d'erreur, pas des mesures. Un **salaire stocké en texte** avec des espaces irréguliers. Et des **catégories incohérentes** — différences de casse et espaces parasites, y compris au milieu des chaînes.

Ce dernier point mérite attention : les doublons ne sont détectés qu'**après** normalisation du texte. Sans cette étape, deux lignes identiques écrites différemment auraient été comptées deux fois.

### Aucune fuite entre entraînement et test

L'encodage et la standardisation sont appris **uniquement sur les données d'entraînement**, via un `Pipeline` scikit-learn. C'est une précaution que beaucoup négligent : standardiser avant de séparer revient à laisser les données de test influencer la moyenne et l'écart-type, et donc à surestimer la performance.

### Multicolinéarité vérifiée avant modélisation

Le VIF de toutes les variables se situe entre **1,00 et 1,01**, très en dessous du seuil de 5. Cette vérification n'est pas décorative : la LDA et la QDA reposent sur l'inversion de matrices de covariance, et une forte colinéarité les rendrait instables. Le résultat **valide l'emploi des deux méthodes génératives**.

### Des hyperparamètres choisis, pas devinés

L'effet du paramètre de régularisation `C` est d'abord étudié visuellement. À forte régularisation (C = 0,001), le modèle est sous-ajusté — l'exactitude sur l'entraînement (0,783) est *inférieure* à celle du test (0,805), signature classique du sous-apprentissage. Quand C augmente, les deux courbes se croisent vers **C = 0,01**.

Le `GridSearchCV` retient indépendamment **C = 0,01** (AUC en validation croisée : 0,851), cohérent avec la lecture graphique. Le paramètre de régularisation de la QDA est déterminé de la même façon.

Le chemin de régularisation L1/L2 confirme par ailleurs la hiérarchie des variables : les heures hebdomadaires et la distance domicile-travail se stabilisent rapidement comme facteurs de risque, le télétravail partiel et la satisfaction comme facteurs protecteurs, tandis que l'âge et le département restent collés à zéro.

---

## Démarche

| Étape | Contenu |
|---|---|
| 1 · Nettoyage | Diagnostic du brut, correction des quatre familles d'anomalies, doublons |
| 2 · Exploration | Probabilités a priori, profils par classe, taux par modalité, corrélations |
| 3 · Modélisation | Pipeline, VIF, effet de C, L1 vs L2, GridSearchCV |
| 4 · Évaluation | Validation croisée, matrices de confusion, courbes ROC comparées |
| 5 · Interprétation | Odds ratios, profils par classe, coefficients LDA, comparaison croisée |
| 6 · Décision | Synthèse, arbitrage métier, recommandations |

---

## Recommandations

1. Surveiller en priorité les employés **cumulant charge horaire élevée et trajet long** — les deux facteurs de risque dominants.
2. Renforcer le suivi de la satisfaction et la politique salariale pour les **employés récents**, l'ancienneté étant fortement protectrice.
3. **Ajuster le seuil de décision** selon le budget RH réellement disponible pour le suivi préventif : à 0,5, le modèle génère beaucoup d'alertes.
4. Ne pas segmenter la politique de rétention par **département ou mode de télétravail** — aucun effet mesurable.

---

## Stack technique

| Domaine | Outils |
|---|---|
| Manipulation | pandas, numpy |
| Préparation | scikit-learn — Pipeline, ColumnTransformer, StandardScaler, OneHotEncoder |
| Modélisation | LogisticRegression, LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis |
| Sélection | GridSearchCV, cross_val_score |
| Diagnostic | statsmodels — variance_inflation_factor |
| Visualisation | matplotlib, seaborn |

---

## Contenu du dépôt

```
attrition-employes/
├── tp_LOGBO_CAMARA.ipynb    # Notebook complet — 6 étapes
├── Presentation.pptx        # Support de soutenance
└── donnees_attrition_brut.csv
```

```bash
pip install pandas numpy scikit-learn statsmodels matplotlib seaborn
jupyter notebook tp_LOGBO_CAMARA.ipynb
```

---

## Limites

Les classes sont déséquilibrées (17,3 % de départs) et l'échantillon est modeste — 1 001 employés, dont environ 173 départs, ce qui limite la précision des estimations sur la classe minoritaire. La précision de 0,38 du modèle retenu signifie que près de deux alertes sur trois ne se concrétiseront pas : le seuil doit être calibré sur la capacité réelle de suivi des équipes RH. Enfin, les variables disponibles décrivent un profil administratif et ne captent ni le climat d'équipe, ni la relation managériale, ni les opportunités externes — des facteurs d'attrition majeurs par ailleurs.

---

## Équipe

**LOGBO Axelle · CAMARA Massaram**

Module *Modèle logistique et analyse discriminante* — Master 1.
