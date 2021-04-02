## Rainbow Memory - Official PyTorch Implementation

<img src="./overview.png" width="450">

**Rainbow Memory: Continual Learning with a Memory of Diverse Samples**<br>
[Paper](https://arxiv.org/pdf/2103.17230.pdf) | [Bibtex](#Citation)<br>
**Jihwan Bang<sup>\*</sup>, Heesu Kim<sup>\*</sup>, YoungJoon Yoo, Jung-Woo Ha, Jonghyun Choi** <br>
CVPR 2021<br>
(\* indicates equal contribution)

**NOTE: The code will be pushed to this repository soon.**

## Abstract
Continual learning is a realistic learning scenario for AI models. 
Prevalent scenario of continual learning, however, assumes disjoint sets of classes as tasks and is less realistic rather artificial. 
Instead, we focus on 'blurry' task boundary; where tasks shares classes and is more realistic and practical. 
To address such task, we argue the importance of diversity of samples in an episodic memory. 
To enhance the sample diversity in the memory, we propose a novel memory management strategy based on per-sample classification uncertainty and data augmentation, named Rainbow Memory (RM). 
With extensive empirical validations on MNIST, CIFAR10, CIFAR100, and ImageNet datasets, 
we show that the proposed method significantly improves the accuracy in blurry continual learning setups, outperforming state of the arts by large margins despite its simplicity.

## Overview of the results of RM
The table is shown for last accuracy comparison in various datasets in Blurry10-Online.
If you want to see more details, see the [paper](https://arxiv.org/pdf/2103.17230.pdf).

| Methods   | MNIST      | CIFAR100   | ImageNet |
|-----------|------------|------------|----------|
| EWC       | 90.98±0.61 | 26.95±0.36 | 39.54    |
| Rwalk     | 90.69±0.62 | 32.31±0.78 | 35.26    |
| iCaRL     | 78.09±0.60 | 17.39±1.04 | 17.52    |
| GDumb     | 88.51±0.52 | 27.19±0.65 | 21.52    |
| BiC       | 77.75±1.27 | 13.01±0.24 | 37.20    |
| **RM w/o DA** | **92.65±0.33** | 34.09±1.41 | 37.96    |
| **RM**        | 91.80±0.69 | **41.35±0.95** | **50.11**    |

## Updates
April 2nd, 2021: Initial upload only README

## Getting Started
TBD.

## Citation 
```angular2
@inproceedings{jihwan2021rainbow,
  title={Rainbow Memory: Continual Learning with a Memory of Diverse Samples},
  author={Jihwan Bang, Heesu Kim, YoungJoon Yoo, Jung-Woo Ha, Jonghyun Choi},
  booktitle={CVPR},
  month={June},
  year={2021}
}
```

## License

```
Copyright (c) 2019-present NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

