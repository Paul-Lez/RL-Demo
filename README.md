# RL-Demo

A small teaching repo for experimenting with the REINFORCE policy-gradient
algorithm on a grid-world task. The main entry point is a notebook that walks
through the environment, policy, one-step updates, training, evaluation, and
trajectory visualizations.

## Contents

- `REINFORCE_gridworld_demo.ipynb` - the walkthrough notebook.
- `reinforce_demo/environment.py` - grid-world dynamics, rewards, obstacles, and actions.
- `reinforce_demo/policies.py` - tabular and MLP softmax policies plus REINFORCE update helpers.
- `reinforce_demo/training.py` - training, evaluation, and rollout sampling loops.
- `reinforce_demo/plotting.py` - plots and animations used by the notebook.
- `reinforce_demo/dashboard.py` and `runs/latest.json` - optional JSON snapshots for live training dashboards.
- `requirements.txt` - Python dependencies.
- `Dockerfile` and `docker-compose.yml` - optional local Jupyter environment.

## Google Colab

Open the notebook in Colab:

https://colab.research.google.com/github/Paul-Lez/RL-Demo/blob/main/REINFORCE_gridworld_demo.ipynb

Then run the cells top to bottom. The first setup cell detects Colab and
clones this repo into `/content/RL-Demo` automatically if needed. A CPU runtime
is enough.

## Local Setup

Clone the repo and install the dependencies from the repo root:

```bash
git clone https://github.com/Paul-Lez/RL-Demo.git
cd RL-Demo
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m jupyter lab REINFORCE_gridworld_demo.ipynb
```

Keep Jupyter running from the repo root so the notebook can import
`reinforce_demo`.

## Docker Option

If you prefer an isolated local environment:

```bash
docker compose up --build
```

Then open `http://127.0.0.1:8888/lab?token=reinforce`.

## Things To Try

In the notebook, change `REWARD_MODE` (`dense`, `sparse`, `sparse_length`,
`dense_noop_penalty`), `POLICY_MODE` (`tabular`, `mlp`, `large_mlp`),
`UPDATE_MODE` (`vanilla`, `advantage`), or `BATCH_SIZE` and compare the learning
curves, learned policy arrows, and rollout animations. Set `USE_GPU=True` to
move the policy network to CUDA or Apple Silicon MPS when available.
