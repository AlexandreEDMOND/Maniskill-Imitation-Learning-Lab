# ManiSkill Imitation Learning Lab

Laboratoire minimal d'imitation learning en robotique avec [ManiSkill](https://maniskill.readthedocs.io/), PyTorch et Behavior Cloning.

La tâche étudiée est `PushCube-v1` : un bras Panda doit pousser un cube sur une table jusqu'à une zone cible. Elle ne demande ni saisie ni levage, ce qui en fait une première tâche adaptée pour valider un pipeline d'imitation en boucle fermée.

## Objectif

À partir de démonstrations expertes ManiSkill, entraîner une policy MLP qui prédit l'action du robot depuis une observation d'état.

```text
observation -> MLP -> action
```

La policy est entraînée par Behavior Cloning avec une erreur quadratique moyenne :

```text
MSE(policy(observation), action_experte)
```

## Contenu du projet

- téléchargement et inspection des démonstrations `PushCube-v1` ;
- conversion optionnelle des démonstrations en observations `state` ;
- chargement des couples `(observation, action)` ;
- normalisation des observations et des actions ;
- entraînement d'une policy MLP simple ;
- évaluation dans ManiSkill et sauvegarde des métriques.

## Installation

Prérequis : Python 3.11+ et `uv`.

```bash
uv sync
uv run python -c "import mani_skill, torch, h5py, gymnasium; print('ok')"
```

## Démarrer

Télécharger les démonstrations officielles :

```bash
uv run python scripts/download_demos.py
find ~/.maniskill/demos/PushCube-v1 -name "*.h5"
```

Inspecter le fichier téléchargé :

```bash
uv run python scripts/inspect_demo.py \
  --demo-path ~/.maniskill/demos/PushCube-v1/motionplanning/trajectory.h5
```

Les démonstrations téléchargées peuvent contenir les états du simulateur (`env_states`) sans les observations `state` directement exploitables. La baseline peut apprendre depuis ces états ; on peut aussi rejouer les trajectoires pour enregistrer les observations de l'environnement.

```bash
uv run python scripts/prepare_state_demos.py \
  --traj-path ~/.maniskill/demos/PushCube-v1/motionplanning/trajectory.h5 \
  --overwrite
```

## Entraîner

Baseline directe à partir des états présents dans les démos :

```bash
uv run python scripts/train_bc.py \
  --demo-path ~/.maniskill/demos/PushCube-v1/motionplanning/trajectory.h5 \
  --observation-source env_states \
  --epochs 50 \
  --checkpoint-path checkpoints/pushcube_bc.pt
```

Après préparation avec `prepare_state_demos.py`, entraîner sur les observations `state` :

```bash
uv run python scripts/train_bc.py \
  --demo-path ~/.maniskill/demos/PushCube-v1/motionplanning/trajectory.state.pd_joint_pos.physx_cpu.h5 \
  --observation-source obs \
  --epochs 50 \
  --checkpoint-path checkpoints/pushcube_bc.pt
```

## Évaluer

```bash
uv run python scripts/evaluate_policy.py \
  --checkpoint-path checkpoints/pushcube_bc.pt \
  --env-id PushCube-v1 \
  --episodes 20 \
  --render-mode none \
  --results-path results/pushcube_eval.json
```

Le JSON produit contient le taux de succès, le retour moyen et le détail de chaque épisode.

## Suivi expérimental

Les mesures ci-dessous utilisent `pd_joint_pos`, des démonstrations ManiSkill
et des observations `env_states`. Les résultats held-out évaluent des seeds qui
ne figurent pas dans le jeu de démonstrations utilisé pour l'entraînement.

| Démos | Méthode | Évaluation | Succès | Retour moyen |
| ---: | --- | --- | ---: | ---: |
| 50 | BC MLP, une action | seeds 0–19 | 1 / 20 (5 %) | 1.94 |
| 1 000 | BC MLP, une action | seeds 0–19 | 3 / 20 (15 %) | 7.16 |
| 1 000 | BC MLP, une action | seeds 1000–1099 held-out | 16 / 100 (16 %) | 7.14 |
| 1 000 | BC MLP, une action, moyenne sur 3 seeds d'entraînement | seeds 1000–1099 held-out | 14.0 % ± 4.4 % | 6.37 ± 0.73 |
| 1 000 | BC MLP, une action, bruit d'observation 0.005 | seeds 1000–1099 held-out | 17 / 100 (17 %) | 6.08 |
| 1 000 | BC MLP, une action, bruit d'observation 0.005, moyenne sur 3 seeds | seeds 1000–1099 held-out | 12.0 % ± 5.0 % | 6.15 ± 0.63 |
| 1 000 | BC MLP, une action, bruit d'observation 0.01 | seeds 1000–1099 held-out | 16 / 100 (16 %) | 6.55 |
| 997 | BC MLP, une action, observations `state` rejouées | seeds 1000–1099 held-out | 4 / 100 (4 %) | 2.43 |
| 1 000 | BC MLP, chunk de 8 actions + moyenne temporelle | seeds 1000–1099 held-out | 0 / 100 (0 %) | 3.21 |
| 1 000 | BC MLP, historique de 4 états, une action | seeds 1000–1099 held-out | 13 / 100 (13 %) | 7.87 |
| 1 000 | BC MLP, historique de 4 états, moyenne sur 3 seeds d'entraînement | seeds 1000–1099 held-out | 12.3 % ± 2.1 % | 6.97 ± 0.83 |

Chaque nouvelle expérience doit être évaluée sur les mêmes 100 seeds held-out
et ajoutée à ce tableau afin de comparer les méthodes à protocole constant. Les
notations ± correspondent à l'écart-type entre les seeds d'entraînement.

### Conclusion actuelle

La référence retenue est le Behavior Cloning MLP qui prédit une action depuis
`env_states`, sans historique ni bruit : `14.0 % ± 4.4 %` de succès held-out.
L'historique de quatre états et le bruit d'observation ne procurent pas de gain
robuste, tandis que le chunking MLP de huit actions échoue complètement. La
prochaine architecture à comparer est donc ACT, qui apprend des séquences
d'actions avec un Transformer et une variable latente, tout en restant en
imitation learning offline.

## Action chunking

La baseline peut prédire plusieurs actions futures depuis une observation. Pendant
l'évaluation, les prédictions disponibles pour l'action courante sont moyennées.
Cette variante a été testée ici et ne dépasse pas la baseline ; elle est conservée
comme point de comparaison pour une future implémentation ACT plus expressive.

```bash
uv run python scripts/train_bc.py \
  --demo-path ~/.maniskill/demos/PushCube-v1/motionplanning/trajectory.h5 \
  --observation-source env_states \
  --max-episodes 1000 \
  --action-horizon 8 \
  --epochs 50 \
  --checkpoint-path checkpoints/pushcube_bc_chunk8_1000demos.pt
```

Pour entraîner avec un historique de quatre observations et une seule action :

```bash
uv run python scripts/train_bc.py \
  --demo-path ~/.maniskill/demos/PushCube-v1/motionplanning/trajectory.h5 \
  --observation-source env_states \
  --max-episodes 1000 \
  --observation-history 4 \
  --action-horizon 1 \
  --epochs 50 \
  --checkpoint-path checkpoints/pushcube_bc_history4_1000demos.pt
```

Pour ajouter un bruit faible aux observations normalisées pendant l'entraînement :

```bash
uv run python scripts/train_bc.py \
  --demo-path ~/.maniskill/demos/PushCube-v1/motionplanning/trajectory.h5 \
  --observation-source env_states \
  --max-episodes 1000 \
  --obs-noise-std 0.005 \
  --epochs 50 \
  --checkpoint-path checkpoints/pushcube_bc_noise005_1000demos.pt
```

## Limites de la baseline

- observations d'état uniquement ;
- MLP sans mémoire temporelle ;
- Behavior Cloning pur, donc sensible aux erreurs accumulées pendant un rollout ;
- aucune donnée de récupération ou DAgger.

## Prochaines étapes

- implémenter et évaluer ACT sur le même protocole held-out ;
- comparer ensuite avec Diffusion Policy ;
- ajouter des observations RGB seulement après une baseline séquentielle state-based ;
- étudier les données de récupération si l'objectif évolue vers une policy plus robuste.
