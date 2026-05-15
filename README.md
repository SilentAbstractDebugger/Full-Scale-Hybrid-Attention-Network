# FSHA-Net: Full-Scale Hybrid Attention Network
## Retinal Vessel Segmentation on DRIVE Dataset

### Install
```bash
pip install torch torchvision albumentations numpy Pillow scipy
```



### Train
```bash
python train.py --model unet    --epochs 50 --batch_size 2
python train.py --model fshanet --epochs 50 --batch_size 2
```

### Evaluate
```bash
python eval.py --model both
```

### Changes from v1
- LR scheduler (ReduceLROnPlateau)
- Early stopping (--patience 15)
- Gradient clipping (--clip_grad 1.0)
- Weight decay 1e-5 in Adam
- HD95 tracked during training validation
- Sensitivity + Specificity added to eval
- Critical analysis printed in comparison table
