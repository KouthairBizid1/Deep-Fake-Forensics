# Shallow-Deepfake Forensics

## Datasets

### Deepfakes

The deepfake dataset we used in Section III.B of our paper can be downloaded [here](https://www.dropbox.com/s/o5410tl5v4vxsth/ICNC2023-Deepfakes.tar.xz?dl=0).

### Shallowfakes

Shallowfake dataset used in our paper can be downloaded separately via the following links:

- [CASIAv2](https://github.com/namtpham/casia2groundtruth)
- [CASIAv1](https://github.com/namtpham/casia1groundtruth)
- [Columbia](https://www.ee.columbia.edu/ln/dvmm/downloads/authsplcuncmp/)
- [COVERAGE](https://github.com/wenbihan/coverage)
- [NIST16](https://www.nist.gov/itl/iad/mig/open-media-forensics-challenge)

### Training/Validation/Testing subsets

The paths of how we split the dataset can be found [here](https://www.dropbox.com/s/opjpz9hoy5xm4um/paths.zip?dl=0).

The format of each line in these files is as the following. For authentic images, `/path/to/mask.png` and `/path/to/egde.png` are set to string `None`. We use digit `0` to represent authentic images, and `1` to represent forged images.

```
/path/to/image.png /path/to/mask.png /path/to/edge.png 0/1
```

## Usage

### Training

Run the following code to train the network.

For the option `--model`, to reproduce experiments in Table III of our paper:
- Use `ours` for experiments 1/2/3.

```
python -u train_torch.py --paths_file /path/to/train.txt --val_paths_file /path/to/val.txt --model ours 
```

### Testing

Run the following code to evaluate the network.

Trained models for experiments in Table I of our paper can be found in the following links: [1](https://drive.google.com/file/d/1AkuBDMnwYY4QJDUetcAqjJ8i00WHl2qm/view?usp=drive_link) | [2](https://drive.google.com/file/d/1j6VU3BeHkxRXB7Rh4UhK6U_D2QQSacAd/view?usp=drive_link) | [3](https://drive.google.com/file/d/1--3Tmng1tr1ZU3N-Kkv6j3z6Zodi85DO/view?usp=drive_link) .
```
python -u evaluate.py --paths_file /path/to/test.txt --load_path /path/to/trained/model.pth --model ours
```