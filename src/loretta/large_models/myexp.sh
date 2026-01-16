# MODEL=facebook/opt-6.7b TASK=CB MODE=lora EPOCH=3 RANK=8 BS=8 LR=1e-4 DEVICE=0 SEED=0  bash finetune.sh
# MODEL=facebook/opt-6.7b TASK=CB MODE=loretta_rep EPOCH=3 RANK=16 BS=8 LR=3e-4 DEVICE=1 SEED=0  bash finetune.sh

# MODEL=facebook/opt-6.7b TASK=BoolQ MODE=lora EPOCH=3 RANK=8 BS=8 LR=1e-4 DEVICE=0 SEED=0  bash finetune.sh
# MODEL=facebook/opt-6.7b TASK=BoolQ MODE=lora EPOCH=3 RANK=16 BS=8 LR=1e-4 DEVICE=0 SEED=0  bash finetune.sh
# MODEL=facebook/opt-6.7b TASK=BoolQ MODE=loretta_rep EPOCH=3 RANK=16 BS=8 LR=1e-4 DEVICE=0 SEED=0  bash finetune.sh

# MODEL=facebook/opt-6.7b TASK=Copa MODE=lora EPOCH=3 RANK=8 BS=8 LR=1e-4 DEVICE=0 SEED=0  bash finetune.sh
# MODEL=facebook/opt-6.7b TASK=Copa MODE=lora EPOCH=3 RANK=16 BS=8 LR=1e-4 DEVICE=0 SEED=0  bash finetune.sh
# MODEL=facebook/opt-6.7b TASK=Copa MODE=loretta_rep EPOCH=3 RANK=16 BS=8 LR=1e-4 DEVICE=0 SEED=0  bash finetune.sh

# MODEL=facebook/opt-6.7b TASK=ReCoRD MODE=lora EPOCH=3 RANK=8 BS=8 LR=1e-4 DEVICE=0 SEED=0  bash finetune.sh
# MODEL=facebook/opt-6.7b TASK=ReCoRD MODE=lora EPOCH=3 RANK=16 BS=8 LR=1e-4 DEVICE=0 SEED=0  bash finetune.sh
# MODEL=facebook/opt-6.7b TASK=ReCoRD MODE=loretta_rep EPOCH=3 RANK=16 BS=8 LR=1e-4 DEVICE=0 SEED=0  bash finetune.sh

# MODEL=facebook/opt-6.7b TASK=SQuAD MODE=lora EPOCH=3 RANK=8 BS=8 LR=1e-4 DEVICE=0 SEED=0  bash finetune.sh
# MODEL=facebook/opt-6.7b TASK=SQuAD MODE=lora EPOCH=3 RANK=16 BS=8 LR=1e-4 DEVICE=0 SEED=0  bash finetune.sh
# MODEL=facebook/opt-6.7b TASK=SQuAD MODE=loretta_rep EPOCH=3 RANK=16 BS=8 LR=1e-4 DEVICE=0 SEED=0  bash finetune.sh

# MODEL=facebook/opt-6.7b TASK=DROP MODE=lora EPOCH=3 RANK=8 BS=8 LR=1e-4 DEVICE=0 SEED=0  bash finetune.sh
# MODEL=facebook/opt-6.7b TASK=DROP MODE=lora EPOCH=3 RANK=16 BS=8 LR=1e-4 DEVICE=0 SEED=0  bash finetune.sh
# MODEL=facebook/opt-6.7b TASK=DROP MODE=loretta_rep EPOCH=3 RANK=16 BS=8 LR=1e-4 DEVICE=0 SEED=0  bash finetune.sh

# MODEL=facebook/opt-6.7b TASK=CB MODE=loretta_rep EPOCH=3 RANK=16 BS=8 LR=1e-4 DEVICE=1 SEED=0 TRAINER=sam SAM_RHO=0.05 bash finetune.sh
# MODEL=facebook/opt-6.7b TASK=CB MODE=loretta_rep EPOCH=3 RANK=16 BS=8 LR=1e-4 DEVICE=1 SEED=0 TRAINER=sam SAM_RHO=0.2 bash finetune.sh

# MODEL=facebook/opt-6.7b TASK=CB MODE=loretta_rep EPOCH=3 RANK=16 BS=8 LR=3e-4 DEVICE=1 SEED=0 TRAINER=sam SAM_RHO=0.05 bash finetune.sh
# MODEL=facebook/opt-6.7b TASK=CB MODE=loretta_rep EPOCH=3 RANK=16 BS=8 LR=3e-4 DEVICE=1 SEED=0 TRAINER=sam SAM_RHO=0.1 bash finetune.sh
# MODEL=facebook/opt-6.7b TASK=CB MODE=loretta_rep EPOCH=3 RANK=16 BS=8 LR=3e-4 DEVICE=1 SEED=0 TRAINER=sam SAM_RHO=0.2 bash finetune.sh

# MODEL=facebook/opt-6.7b TASK=CB MODE=loretta_rep EPOCH=3 RANK=16 BS=8 LR=5e-4 DEVICE=1 SEED=0 TRAINER=sam SAM_RHO=0.05 bash finetune.sh
# MODEL=facebook/opt-6.7b TASK=CB MODE=loretta_rep EPOCH=3 RANK=16 BS=8 LR=5e-4 DEVICE=1 SEED=0 TRAINER=sam SAM_RHO=0.1 bash finetune.sh
# MODEL=facebook/opt-6.7b TASK=CB MODE=loretta_rep EPOCH=3 RANK=16 BS=8 LR=5e-4 DEVICE=1 SEED=0 TRAINER=sam SAM_RHO=0.2 bash finetune.sh

# MODEL=facebook/opt-6.7b TASK=Copa MODE=loretta_rep EPOCH=3 RANK=16 BS=8 LR=1e-4 DEVICE=0 SEED=0 TRAINER=gbar GBAR_ALPHA=0.5 bash finetune.sh
# MODEL=facebook/opt-6.7b TASK=Copa MODE=loretta_rep EPOCH=3 RANK=16 BS=8 LR=1e-4 DEVICE=0 SEED=0 TRAINER=gbar GBAR_ALPHA=1 bash finetune.sh
# MODEL=facebook/opt-6.7b TASK=Copa MODE=loretta_rep EPOCH=3 RANK=16 BS=8 LR=1e-4 DEVICE=0 SEED=0 TRAINER=gbar GBAR_ALPHA=2 bash finetune.sh

# MODEL=facebook/opt-6.7b TASK=Copa MODE=loretta_rep EPOCH=3 RANK=16 BS=8 LR=1e-4 DEVICE=0 SEED=0 TRAINER=sam SAM_RHO=0.05 bash finetune.sh
# MODEL=facebook/opt-6.7b TASK=Copa MODE=loretta_rep EPOCH=3 RANK=16 BS=8 LR=1e-4 DEVICE=0 SEED=0 TRAINER=sam SAM_RHO=0.1 bash finetune.sh
# MODEL=facebook/opt-6.7b TASK=Copa MODE=loretta_rep EPOCH=3 RANK=16 BS=8 LR=1e-4 DEVICE=0 SEED=0 TRAINER=sam SAM_RHO=0.2 bash finetune.sh

TASK=$1
BS=$2
# MODEL=facebook/opt-6.7b TASK=$TASK MODE=lora EPOCH=3 RANK=8 BS=$BS LR=1e-4 DEVICE=$3 SEED=0  bash finetune.sh
# MODEL=facebook/opt-6.7b TASK=$TASK MODE=lora EPOCH=3 RANK=8 BS=$BS LR=3e-4 DEVICE=$3 SEED=0  bash finetune.sh
# MODEL=facebook/opt-6.7b TASK=$TASK MODE=lora EPOCH=3 RANK=8 BS=$BS LR=5e-4 DEVICE=$3 SEED=0  bash finetune.sh

# MODEL=facebook/opt-6.7b TASK=$TASK MODE=lora EPOCH=3 RANK=16 BS=$BS LR=1e-4 DEVICE=$3 SEED=0  bash finetune.sh
# MODEL=facebook/opt-6.7b TASK=$TASK MODE=lora EPOCH=3 RANK=16 BS=$BS LR=3e-4 DEVICE=$3 SEED=0  bash finetune.sh
# MODEL=facebook/opt-6.7b TASK=$TASK MODE=lora EPOCH=3 RANK=16 BS=$BS LR=5e-4 DEVICE=$3 SEED=0  bash finetune.sh

# MODEL=facebook/opt-6.7b TASK=$TASK MODE=loretta_rep EPOCH=3 RANK=16 BS=$BS LR=1e-4 DEVICE=$3 SEED=0  bash finetune.sh
# MODEL=facebook/opt-6.7b TASK=$TASK MODE=loretta_rep EPOCH=3 RANK=16 BS=$BS LR=3e-4 DEVICE=$3 SEED=0  bash finetune.sh
# MODEL=facebook/opt-6.7b TASK=$TASK MODE=loretta_rep EPOCH=3 RANK=16 BS=$BS LR=5e-4 DEVICE=$3 SEED=0  bash finetune.sh

# MODEL=facebook/opt-6.7b TASK=$TASK MODE=loretta_rep EPOCH=3 RANK=16 BS=$BS LR=1e-4 DEVICE=$3 SEED=0 TRAINER=sam SAM_RHO=0.05 bash finetune.sh
# MODEL=facebook/opt-6.7b TASK=$TASK MODE=loretta_rep EPOCH=3 RANK=16 BS=$BS LR=1e-4 DEVICE=$3 SEED=0 TRAINER=sam SAM_RHO=0.1 bash finetune.sh
# MODEL=facebook/opt-6.7b TASK=$TASK MODE=loretta_rep EPOCH=3 RANK=16 BS=$BS LR=1e-4 DEVICE=$3 SEED=0 TRAINER=sam SAM_RHO=0.2 bash finetune.sh

# MODEL=facebook/opt-6.7b TASK=$TASK MODE=loretta_rep EPOCH=3 RANK=16 BS=$BS LR=3e-4 DEVICE=$3 SEED=0 TRAINER=sam SAM_RHO=0.05 bash finetune.sh
# MODEL=facebook/opt-6.7b TASK=$TASK MODE=loretta_rep EPOCH=3 RANK=16 BS=$BS LR=3e-4 DEVICE=$3 SEED=0 TRAINER=sam SAM_RHO=0.1 bash finetune.sh
# MODEL=facebook/opt-6.7b TASK=$TASK MODE=loretta_rep EPOCH=3 RANK=16 BS=$BS LR=3e-4 DEVICE=$3 SEED=0 TRAINER=sam SAM_RHO=0.2 bash finetune.sh

# MODEL=facebook/opt-6.7b TASK=$TASK MODE=loretta_rep EPOCH=3 RANK=16 BS=$BS LR=5e-4 DEVICE=$3 SEED=0 TRAINER=sam SAM_RHO=0.05 bash finetune.sh
# MODEL=facebook/opt-6.7b TASK=$TASK MODE=loretta_rep EPOCH=3 RANK=16 BS=$BS LR=5e-4 DEVICE=$3 SEED=0 TRAINER=sam SAM_RHO=0.1 bash finetune.sh
# MODEL=facebook/opt-6.7b TASK=$TASK MODE=loretta_rep EPOCH=3 RANK=16 BS=$BS LR=5e-4 DEVICE=$3 SEED=0 TRAINER=sam SAM_RHO=0.2 bash finetune.sh

MODEL=facebook/opt-6.7b TASK=$TASK MODE=loretta_rep EPOCH=3 RANK=16 BS=$BS LR=1e-4 DEVICE=$3 SEED=0 TRAINER=gbar2 GBAR_ALPHA=0.5 bash finetune.sh
# MODEL=facebook/opt-6.7b TASK=$TASK MODE=loretta_rep EPOCH=3 RANK=16 BS=$BS LR=1e-4 DEVICE=$3 SEED=0 TRAINER=gbar2 GBAR_ALPHA=0.1 bash finetune.sh
# MODEL=facebook/opt-6.7b TASK=$TASK MODE=loretta_rep EPOCH=3 RANK=16 BS=$BS LR=1e-4 DEVICE=$3 SEED=0 TRAINER=gbar2 GBAR_ALPHA=0.05 bash finetune.sh

# MODEL=facebook/opt-6.7b TASK=$TASK MODE=loretta_rep EPOCH=3 RANK=16 BS=$BS LR=3e-4 DEVICE=$3 SEED=0 TRAINER=gbar2 GBAR_ALPHA=0.5 bash finetune.sh
# MODEL=facebook/opt-6.7b TASK=$TASK MODE=loretta_rep EPOCH=3 RANK=16 BS=$BS LR=3e-4 DEVICE=$3 SEED=0 TRAINER=gbar2 GBAR_ALPHA=0.1 bash finetune.sh
# MODEL=facebook/opt-6.7b TASK=$TASK MODE=loretta_rep EPOCH=3 RANK=16 BS=$BS LR=3e-4 DEVICE=$3 SEED=0 TRAINER=gbar2 GBAR_ALPHA=0.05 bash finetune.sh

# MODEL=facebook/opt-6.7b TASK=$TASK MODE=loretta_rep EPOCH=3 RANK=16 BS=$BS LR=5e-4 DEVICE=$3 SEED=0 TRAINER=gbar2 GBAR_ALPHA=0.5 bash finetune.sh
# MODEL=facebook/opt-6.7b TASK=$TASK MODE=loretta_rep EPOCH=3 RANK=16 BS=$BS LR=5e-4 DEVICE=$3 SEED=0 TRAINER=gbar2 GBAR_ALPHA=0.1 bash finetune.sh
# MODEL=facebook/opt-6.7b TASK=$TASK MODE=loretta_rep EPOCH=3 RANK=16 BS=$BS LR=5e-4 DEVICE=$3 SEED=0 TRAINER=gbar2 GBAR_ALPHA=0.05 bash finetune.sh

# MODEL=facebook/opt-6.7b TASK=CB MODE=ft EPOCH=0 BS=1 LR=4e-6 DEVICE=$1 SEED=0 TRAINER=none bash finetune.sh
# MODEL=facebook/opt-6.7b TASK=BoolQ MODE=ft EPOCH=0 BS=2 LR=4e-6 DEVICE=$1 SEED=0 TRAINER=none bash finetune.sh
# MODEL=facebook/opt-6.7b TASK=WSC MODE=ft EPOCH=0 BS=2 LR=4e-6 DEVICE=$1 SEED=0 TRAINER=none bash finetune.sh
# MODEL=facebook/opt-6.7b TASK=Copa MODE=ft EPOCH=0 BS=2 LR=4e-6 DEVICE=$1 SEED=0 TRAINER=none bash finetune.sh
# MODEL=facebook/opt-6.7b TASK=ReCoRD MODE=ft EPOCH=0 BS=2 LR=4e-6 DEVICE=$1 SEED=0 TRAINER=none bash finetune.sh
# MODEL=facebook/opt-6.7b TASK=SQuAD MODE=ft EPOCH=0 BS=1 LR=4e-6 DEVICE=$1 SEED=0 TRAINER=none bash finetune.sh
# MODEL=facebook/opt-6.7b TASK=DROP MODE=ft EPOCH=0 BS=1 LR=4e-6 DEVICE=$1 SEED=0 TRAINER=none bash finetune.sh
