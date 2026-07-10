# Maniskill-Imitation-Learning-Lab

Premier laboratoire d'imitation learning en robotique avec [ManiSkill](https://maniskill.readthedocs.io/), PyTorch et une policy MLP simple.

Le projet commence volontairement par une tâche simple, `PickCube-v1`, avec des observations state-based. L'objectif est de comprendre toute la chaîne avant d'ajouter des images RGB, des architectures plus fortes ou des tâches custom.

## Ce que fait ce projet

- télécharge les démonstrations ManiSkill de `PickCube-v1` ;
- inspecte les fichiers `.h5` pour comprendre leur structure ;
- prépare des démonstrations avec observations `state` si nécessaire ;
- peut entraîner directement depuis les `env_states` déjà présents dans les démonstrations ;
- charge les couples `(observation, action)` ;
- entraîne une policy de Behavior Cloning avec une loss MSE ;
- évalue la policy dans ManiSkill ;
- sauvegarde checkpoints et métriques d'évaluation.

## Imitation learning

L'imitation learning consiste à entraîner une policy à reproduire le comportement d'un expert à partir de démonstrations. Ici, on utilise du Behavior Cloning : le modèle reçoit une observation et apprend à prédire l'action expert correspondante.

Formellement, on minimise :

```text
MSE(policy(observation), expert_action)
```

C'est une baseline simple, utile pour valider le pipeline de données, l'environnement et les métriques avant de passer à des méthodes plus avancées.

## Pourquoi PickCube-v1

`PickCube-v1` est une bonne première tâche parce qu'elle est courte, visuelle et facile à raisonner : le robot doit saisir un cube et le déplacer vers une cible. Elle expose déjà les difficultés classiques de la robotique, comme le contrôle continu, le contact et la généralisation, sans imposer une architecture complexe dès le départ.

## Installation

Prérequis :

- Python 3.11+
- `uv`
- un environnement capable d'installer PyTorch et ManiSkill

Depuis la racine du repo :

```bash
uv sync
```

Si votre cache global `uv` a un problème de permissions, utilisez un cache local au repo :

```bash
UV_CACHE_DIR=.uv-cache uv sync
```

Vérifier l'installation :

```bash
uv run python -c "import mani_skill, torch, h5py, gymnasium; print('ok')"
```

## Télécharger les démonstrations

ManiSkill fournit un utilitaire de téléchargement :

```bash
uv run python scripts/download_demos.py --env-id PickCube-v1
```

Les fichiers sont généralement placés sous :

```text
~/.maniskill/demos/PickCube-v1/
```

Les démonstrations téléchargées peuvent être en `obs_mode=none`. Dans ce cas, elles contiennent les états du simulateur, mais pas directement les observations `state` ManiSkill. Deux options existent :

- entraîner directement sur les `env_states` stockés dans le `.h5`, ce qui ne lance pas SAPIEN ;
- rejouer les trajectoires pour créer un fichier avec observations `state`, ce qui nécessite que ManiSkill/SAPIEN puisse créer l'environnement localement.

Sur macOS, si SAPIEN échoue avec une erreur Vulkan ou `vk::createInstanceUnique`, utilisez d'abord le mode `env_states`.

Exemple recommandé avec les démonstrations `motionplanning` :

```bash
uv run python scripts/prepare_state_demos.py \
  --traj-path ~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.h5 \
  --output-name trajectory.state.pd_joint_pos.physx_cpu.h5 \
  --num-envs 1
```

Pour un test rapide, ajoutez `--count 20` afin de ne rejouer que 20 épisodes.

Si le nom du fichier téléchargé est différent, inspectez le dossier :

```bash
find ~/.maniskill/demos/PickCube-v1 -name "*.h5"
```

## Inspecter les données

Avant d'entraîner, inspectez le fichier `.h5` :

```bash
uv run python scripts/inspect_demo.py \
  --demo-path ~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.state.pd_joint_pos.physx_cpu.h5
```

Le script affiche :

- les groupes de trajectoires ;
- la présence de `obs` et `actions` ;
- les shapes et dtypes ;
- un diagnostic clair si le fichier n'est pas prêt pour le Behavior Cloning state-based.

## Entraîner le modèle

Option robuste sans replay SAPIEN, directement depuis les démonstrations téléchargées :

```bash
uv run python scripts/train_bc.py \
  --demo-path ~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.h5 \
  --observation-source env_states \
  --epochs 50 \
  --batch-size 256 \
  --checkpoint-path checkpoints/pickcube_bc_env_states.pt
```

Option après conversion en `obs_mode=state` :

```bash
uv run python scripts/train_bc.py \
  --demo-path ~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.state.pd_joint_pos.physx_cpu.h5 \
  --epochs 50 \
  --batch-size 256 \
  --checkpoint-path checkpoints/pickcube_bc.pt
```

Le modèle est un MLP simple. Il prend une observation aplatie et prédit une action continue. Les checkpoints sont sauvegardés dans `checkpoints/`.

## Évaluer la policy

```bash
uv run python scripts/evaluate_policy.py \
  --checkpoint-path checkpoints/pickcube_bc.pt \
  --env-id PickCube-v1 \
  --episodes 20 \
  --results-path results/pickcube_eval.json
```

Le script sauvegarde un fichier JSON avec les retours, longueurs d'épisode et taux de succès si l'environnement expose cette information.

## Limitations actuelles

- observations state-based uniquement ;
- policy MLP sans historique temporel ;
- pas de normalisation des observations/actions ;
- pas d'augmentation de données ;
- pas de rendu RGB pendant l'entraînement ;
- performance attendue limitée par la simplicité du Behavior Cloning.

## État expérimental

Validé :

- expert replay : `success_rate = 1.0` ;
- nearest-neighbor sur une démonstration : succès online ;
- BC vanilla : erreur offline quasi nulle, mais échec online ;
- BC-v2-b (`state + timestep + previous_action`) : succès closed-loop sur `traj_0` ;
- BC sur 10 démonstrations : imitation offline excellente.

L'évaluation online des diagnostics 10-démo doit être lancée avec MoltenVK explicite sur macOS :

```bash
VK_ICD_FILENAMES=/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json uv run python -c \
  "import gymnasium as gym; import mani_skill.envs; env = gym.make('PickCube-v1', obs_mode='state', control_mode='pd_joint_pos'); print('env ok'); env.close()"
```

Avec SAPIEN 3.0.3, `vulkaninfo` peut détecter MoltenVK alors que SAPIEN échoue encore sans cette variable.

## Roadmap

- Behavior Cloning avec observations RGB ;
- Diffusion Policy ;
- ACT ;
- comparaison avec PPO ;
- tâche custom ManiSkill ;
- lien futur avec LeRobot / SO-101.

## Commandes exactes depuis un repo vide

```bash
git clone https://github.com/AlexandreEDMOND/Maniskill-Imitation-Learning-Lab.git
cd Maniskill-Imitation-Learning-Lab

UV_CACHE_DIR=.uv-cache uv sync

uv run python scripts/download_demos.py --env-id PickCube-v1

find ~/.maniskill/demos/PickCube-v1 -name "*.h5"

uv run python scripts/prepare_state_demos.py \
  --traj-path ~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.h5 \
  --output-name trajectory.state.pd_joint_pos.physx_cpu.h5 \
  --num-envs 1 \
  --count 20

uv run python scripts/inspect_demo.py \
  --demo-path ~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.state.pd_joint_pos.physx_cpu.h5

uv run python scripts/train_bc.py \
  --demo-path ~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.h5 \
  --observation-source env_states \
  --epochs 50 \
  --batch-size 256 \
  --checkpoint-path checkpoints/pickcube_bc_env_states.pt

uv run python scripts/evaluate_policy.py \
  --checkpoint-path checkpoints/pickcube_bc.pt \
  --env-id PickCube-v1 \
  --episodes 20 \
  --results-path results/pickcube_eval.json
```
