#!/bin/bash
# Create the directory layout the experiment scripts assume.
#
# The scripts under experiments/ were written for two machines and refer to fixed paths:
#   on the cluster master:  /home/ubuntu/OverlayCCL  (this repo)      /home/ubuntu/g2 (the gradient-problem harnesses)
#   on the analysis host:   ~/work/paper/overlayccl_public (this repo)  /tmp/g2 (same harnesses, results under /tmp/g2/hw)
# Rather than rewrite every script, this creates symlinks so a fresh clone satisfies both layouts.
# Usage: bash reproduce/setup_layout.sh [master|analysis]   (default: analysis)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
NP=$ROOT/experiments/cross_scope_2026_10_26/new_problems
MODE=${1:-analysis}
link() { local target=$1 name=$2; mkdir -p "$(dirname "$name")"; if [ -e "$name" ] && [ ! -L "$name" ]; then echo "exists, not a link: $name (leave as is)"; else ln -sfn "$target" "$name"; echo "$name -> $target"; fi; }
if [ "$MODE" = master ]; then
  link "$ROOT" /home/ubuntu/OverlayCCL
  link "$NP" /home/ubuntu/g2
  mkdir -p "$NP/amp7/json" "$NP/newprobs"
  echo "write /home/ubuntu/venv.env with: VENV=<python venv with torch-neuronx>  WORKERS=\"<worker private IPs, space separated>\""
  echo "export SSH_KEY=<private key that reaches the workers> (the scripts use \$SSH_KEY for ssh/scp/rsync)"
else
  link "$ROOT" "$HOME/work/paper/overlayccl_public"
  link "$NP" /tmp/g2
  link "$NP/results" "$NP/hw"
  echo "simulator scoring needs a CPU-only venv at ~/work/ncc/venv (python3 -m venv ~/work/ncc/venv && pip install torch numpy anthropic)"
fi
