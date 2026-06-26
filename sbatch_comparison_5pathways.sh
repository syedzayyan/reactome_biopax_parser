#!/bin/bash
#SBATCH -J comparison_5pathways
#SBATCH -o %x.o%j
#SBATCH -p andrena
#SBATCH -A pilot_andrena
#SBATCH -n 12
#SBATCH --cpus-per-gpu=12
#SBATCH -t 72:0:0
#SBATCH --mem-per-cpu=7500M
#SBATCH --gres=gpu:1

cd reactome_biopax_parser
source .venv/bin/activate

# Tunes + benchmarks RGCN, TGAT and STHN on all 5 pathways (Immune, Cell
# Cycle, Hemostasis, Metabolism, Gene Expression) for a direct three-way
# comparison. --time-ablation adds the no-time-encoding variants for both
# TGAT and STHN (RGCN always runs both variants regardless). Results land
# under results/comparison/<pathway_slug>/.
./run_comparison_5pathways.sh --gpu --hits --time-ablation
