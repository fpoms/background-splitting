# Background Splitting: Finding Rare Classes in a Sea of Background

This code provides a PyTorch implementation and pretrained models for Background Splitting, as described in the paper [Background Splitting: Finding Rare Classes in a Sea of Background](https://openaccess.thecvf.com/content/CVPR2021/html/Mullapudi_Background_Splitting_Finding_Rare_Classes_in_a_Sea_of_Background_CVPR_2021_paper.html).

## Train

```
python3 main.py \
        --dataset-type inaturalist17 \
        --data-root /path/to/inaturalist \
        --data-split ./data_splits/inaturalist17_bg_100.txt \
        --image-input-size 299 \
        --val-frequency 1 \
        --checkpoint-frequency 1 \
        --output-dir /path/to/checkpoint_dir
```

## Citation

```
@InProceedings{Mullapudi_2021_CVPR,
    author    = {Mullapudi, Ravi Teja and Poms, Fait and Mark, William R. and Ramanan, Deva and Fatahalian, Kayvon},
    title     = {Background Splitting: Finding Rare Classes in a Sea of Background},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {8043-8052}
}
```
