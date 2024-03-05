# Train YOWOv2 on UCF24 dataset
python train.py \
        --cuda \
        -d ucf24 \
        -v yowo_v2_medium \
        --tfboard \
        -r /home/minhtran/works/HAR/YOWOv2/weights/ucf24/yowo_v2_medium/yowo_v2_medium_epoch_15.pth \
        --root /home/minhtran/works/HAR/YOWOv2/data \
        --num_workers 4 \
        --eval_epoch 1 \
        --max_epoch 20 \
        --lr_epoch 2 3 4 5 \
        -lr 0.0001 \
        -ldr 0.5 \
        -bs 16 \
        -accu 16 \
        -K 16 \
        # --eval \
