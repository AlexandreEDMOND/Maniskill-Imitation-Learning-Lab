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

## Limites de la baseline

- observations d'état uniquement ;
- MLP sans mémoire temporelle ;
- Behavior Cloning pur, donc sensible aux erreurs accumulées pendant un rollout ;
- aucune donnée de récupération ou DAgger.

## Prochaines étapes

- vérifier le replay des démonstrations et la baseline BC ;
- comparer avec des observations RGB ;
- ajouter des données de récupération ;
- comparer avec une méthode séquentielle, comme ACT ou Diffusion Policy.
