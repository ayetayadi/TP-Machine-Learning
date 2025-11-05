# 📘 Travaux Pratiques de Machine Learning  
**ISAMM – Université de la Manouba**  
**Encadré par : Dr. Mohamed Ridha Amamou**  
**Réalisé par : Ayet Ayadi, Chaima Hermi – 3INLOG1**  
**Année universitaire : 2025–2026**

---

## 1. Objectif global du module
Ce projet regroupe l’ensemble des **travaux pratiques du module de Machine Learning**, permettant de maîtriser les algorithmes fondamentaux d’apprentissage supervisé et non supervisé, depuis la régression linéaire jusqu’à la détection d’anomalies.

Chaque TP met l’accent sur :
- la compréhension théorique des modèles ;
- leur implémentation en Python (avec `NumPy`, `Pandas`, `Matplotlib`, `scikit-learn`) ;
- l’interprétation des résultats et la validation des performances.

---

## 2. Contenu des TPs

### 🧩 TP1 — Régression Linéaire et Descente de Gradient
**Objectif :**  
Implémenter et analyser la descente de gradient pour ajuster les paramètres d’un modèle de régression linéaire.

**Points clés :**
- Implémentation du gradient descendant avec normalisation des données.  
- Étude de l’effet des hyperparamètres (`α`, nombre d’itérations).  
- Correction d’une anomalie de prétraitement (normalisation manquante).  
- Extension vers un **modèle quadratique** pour capturer les non-linéarités.  

**Résultats :**
- Réduction de la MAE après normalisation et bonne initialisation.  
- Le modèle quadratique présente une meilleure précision que le modèle linéaire.

**Remarques :**
- La normalisation de toutes les variables est indispensable.  
- Un bon choix du pas d’apprentissage améliore la convergence.

---

### 🔍 TP2 — K-Nearest Neighbors (KNN) et Régression Linéaire
**Objectif :**  
Appliquer des méthodes supervisées pour la classification et la régression.

**Méthodes :**
- Implémentation de `KNeighborsClassifier` et `KNeighborsRegressor` de `scikit-learn`.  
- Comparaison entre distances **Euclidienne** et **Manhattan**.  
- Application sur les datasets *Forest CoverType* (classification) et *Diabetes* (régression).  
- Développement d’un modèle de régression linéaire pour prédire le prix d’appartements.

**Résultats :**
- Précision stable pour KNN, légère amélioration avec distance Manhattan.  
- Régression linéaire convergente avec un coût final minimal (~1.2×10⁻⁴).  
- Le modèle final :  
  \[
  \hat{y} = 1000 \times (0.1069 + 0.1301 \frac{Surface}{500} - 0.1546 \frac{Etage}{10} + 0.2517 \frac{Nb\_pieces}{10})
  \]

**Remarques :**
- Le choix de `k` et du type de distance influence fortement la performance.  
- La normalisation garantit une meilleure stabilité des résultats.

---

### 🌀 TP3.1 — K-Means : Implémentation et Évaluation
**Objectif :**  
Mettre en œuvre l’algorithme de groupement K-Means pour comprendre la logique de regroupement et la convergence des centroïdes.

**Étapes :**
- Implémentation manuelle du K-Means.  
- Comparaison avec l’implémentation `KMeans` de `scikit-learn`.  
- Évaluation des clusters selon la compacité (W) et la séparation (B).  
- Comparaison entre distances Euclidienne et Manhattan.

**Résultats :**
| Distance | W (Intra) | B (Inter) | B/W |
|-----------|------------|-----------|------|
| Euclidienne | 1.731 | 1.402 | 0.810 |
| Manhattan | 1.776 | 1.357 | 0.764 |

**Conclusion :**
- Les clusters Euclidiens sont plus compacts et mieux séparés.  
- L’implémentation manuelle donne des résultats identiques à `scikit-learn`.

---

### ⚙️ TP3.2 — Détection d’Anomalies et Validation par Clustering
**Objectif :**  
Détecter des comportements atypiques à partir d’un modèle statistique et géométrique combiné.

**Méthodes :**
1. **Modélisation statistique :**  
   - Calcul des moyennes, variances et matrice de covariance.  
   - Application du modèle de densités partielles (Gaussiennes indépendantes).  
2. **Détection d’anomalies :**  
   - Seuil fixe : ε = 0.02  
   - Aucune anomalie détectée (toutes les probabilités p(x) > ε).  
3. **Validation géométrique :**  
   - Utilisation du K-Means pour vérifier l’appartenance aux clusters via le rayon intra-cluster.

**Résultats :**
- Tous les nouveaux individus sont acceptés.  
- La base consolidée reste statistiquement stable et cohérente.

**Remarques :**
- Combiner la probabilité et la distance renforce la robustesse du modèle.  
- Cette approche hybride est particulièrement adaptée à la détection d’anomalies comportementales.

---

## 3. Outils utilisés
- **Langage :** Python 3.11  
- **Bibliothèques :**
  - `NumPy`, `Pandas`, `Matplotlib`
  - `scikit-learn`
- **Environnement de travail :**
  - Jupyter Notebook (`.ipynb`)
  - LaTeX (rapport académique)
  - Visualisation avec Matplotlib et DataFrames

---

## 4. Synthèse générale
Ces travaux ont permis de :
- Comprendre les mécanismes fondamentaux des modèles d’apprentissage supervisé et non supervisé.  
- Expérimenter le processus complet d’un projet de Machine Learning : **préparation, apprentissage, évaluation, interprétation**.  
- Développer des compétences techniques (implémentation, visualisation, analyse) et analytiques (interprétation, choix des hyperparamètres).  

Les résultats obtenus démontrent une progression claire vers la maîtrise des techniques de modélisation et d’analyse de données, constituant une base solide pour des études plus avancées en intelligence artificielle.

---

## 5. Structure du projet

