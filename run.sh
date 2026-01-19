#!/bin/bash

#SBATCH --partition=gpu_min80gb
#SBATCH --qos=gpu_min80gb
#SBATCH --job-name=uni
#SBATCH --output=log/slurm_%x.%j.out
#SBATCH --error=log/slurm_%x.%j.err

DATASETS="FGNET SMEAR2005 HCI CAR NEW_THYROID ABALONE5 ABALONE10 BALANCE_SCALE"

LRS="1e-3 1e-4 5e-4 1e-2"
MODELS="snn_resnet18 ""
REPS=`seq 1 4`
MAX_PARALLEL=4

LOSSES="CrossEntropy POM OrdinalEncoding CrossEntropy_UR CDW_CE BinomialUnimodal_CE PoissonUnimodal UnimodalNet ORD_ACL VS_SL"

# Losses that require lambda
LOSSES_LAMBDA="WassersteinUnimodal_KLDIV WassersteinUnimodal_Wass CO2 HO2"

LAMDAS="0.1 1 10 100"

mkdir -p output
mkdir -p log

echo "========================================="
echo "Experiment Parameters:"
echo "========================================="
echo "Datasets: $DATASETS"
echo "Models: $MODELS"
echo "Learning rates: $LRS"
echo "Losses: $LOSSES"
echo "Losses with lambda: $LOSSES_LAMBDA"
echo "Lambda values: $LAMDAS"
echo "Repetitions: $REPS"
echo "========================================="
echo "Starting experiments at: $(date)"
echo "========================================="

job_count=0

# Function: wait until the number of running jobs is less than MAX_PARALLEL
wait_for_jobs() {
    while [ $(jobs -r | wc -l) -ge $MAX_PARALLEL ]; do
        sleep 1
    done
}


# MAIN LOOP
for DATASET in $DATASETS; do
    for MODEL in $MODELS; do
        for LR in $LRS; do
            echo "Processing: Dataset=$DATASET, Model=$MODEL, LR=$LR"
            
            # process losses that do not require lambda
            for LOSS in $LOSSES; do
                for REP in $REPS; do
                    wait_for_jobs  
                    
                    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")_${RANDOM}
                    NAME="output/model_${TIMESTAMP}_${DATASET}_${LOSS}_${MODEL}_${LR}_${REP}.pth"
                    
                    echo "Starting: $LOSS REP=$REP [Job $((++job_count))]"
                    python main.py \
                        --dataset "$DATASET" \
                        --loss "$LOSS" \
                        --REP "$REP" \
                        --output "$NAME" \
                        --batchsize 32 \
                        --epochs 300 \
                        --lr "$LR" \
                        --model "$MODEL" \
                        --learnable_params True \
                        --use_original_loss True &
                done
            done
            
            # process losses that require lambda
            for LOSS in $LOSSES_LAMBDA; do
                for LAMDA in $LAMDAS; do
                    wait_for_jobs  
                    
                    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")_${RANDOM}
                    NAME="output/model_${TIMESTAMP}_${DATASET}_${LOSS}_${MODEL}_${LR}_${LAMDA}_0.pth"
                    
                    echo "Starting: $LOSS Lambda=$LAMDA [Job $((++job_count))]"
                    python main.py \
                        --dataset "$DATASET" \
                        --loss "$LOSS" \
                        --REP "$REP"  \
                        --output "$NAME" \
                        --batchsize 32 \
                        --lamda "$LAMDA" \
                        --epochs 300 \
                        --lr "$LR" \
                        --model "$MODEL" \
                        --learnable_params True \
                        --use_original_loss True &
                done
            done
        done
    done
done


wait
echo "All $job_count jobs completed!"


echo "========================================="
echo "All experiments completed at: $(date)"
echo "========================================="


echo "Checking output files..."
TOTAL_FILES=$(find output -name "*.pth" | wc -l)
echo "Total output files found: $TOTAL_FILES"
echo "========================================"