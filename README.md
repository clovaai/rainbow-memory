#### (NOTICE) The code will be pushed to this repository soon.  
## Rainbow Memory: Continual Learning with a Memory of Diverse Samples
Official PyTorch Implementation of Rainbow Memory | [Paper](https://arxiv.org/pdf/2103.17230.pdf) | [Bibtex](#Citation) 

**Jihwan Bang<sup>\*</sup>, Heesu Kim<sup>\*</sup>, YoungJoon Yoo, Jung-Woo Ha, Jonghyun Choi**

\* indicates equal contribution.

In CVPR 2021.

![overview](./overview.png)

## Abstract
Continual learning is a realistic learning scenario for AI models. 
Prevalent scenario of continual learning, however, assumes disjoint sets of classes as tasks and is less realistic rather artificial. 
Instead, we focus on 'blurry' task boundary; where tasks shares classes and is more realistic and practical. 
To address such task, we argue the importance of diversity of samples in an episodic memory. 
To enhance the sample diversity in the memory, we propose a novel memory management strategy based on per-sample classification uncertainty and data augmentation, named Rainbow Memory (RM). 
With extensive empirical validations on MNIST, CIFAR10, CIFAR100, and ImageNet datasets, 
we show that the proposed method significantly improves the accuracy in blurry continual learning setups, outperforming state of the arts by large margins despite its simplicity.

## Results 
### Various Datasets
<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-0lax{text-align:left;vertical-align:top}
</style>
<table class="tg">
<thead>
  <tr>
    <th class="tg-0lax"></th>
    <th class="tg-0lax" colspan="3">MNIST (K=500)</th>
    <th class="tg-0lax" colspan="3">CIFAR100 (K=2,000)</th>
    <th class="tg-0lax" colspan="3">ImageNet (K=20,000)</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0lax">Methods</td>
    <td class="tg-0lax">A5(↑)</td>
    <td class="tg-0lax">F5(↓)</td>
    <td class="tg-0lax">I5(↓)</td>
    <td class="tg-0lax">A5(↑)</td>
    <td class="tg-0lax"><span style="font-weight:400;font-style:normal">F5(↓)</span></td>
    <td class="tg-0lax"><span style="font-weight:400;font-style:normal">I5(↓)</span></td>
    <td class="tg-0lax">A5(↑)</td>
    <td class="tg-0lax"><span style="font-weight:400;font-style:normal">F5(↓)</span></td>
    <td class="tg-0lax"><span style="font-weight:400;font-style:normal">I5(↓)</span></td>
  </tr>
  <tr>
    <td class="tg-0lax">EWC</td>
    <td class="tg-0lax">90.98/0.61 </td>
    <td class="tg-0lax">4.23/0.45</td>
    <td class="tg-0lax">4.54/0.94</td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
  </tr>
  <tr>
    <td class="tg-0lax">Rwalk</td>
    <td class="tg-0lax">90.69/0.62</td>
    <td class="tg-0lax">4.77/0.36</td>
    <td class="tg-0lax">4.96/0.56</td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
  </tr>
  <tr>
    <td class="tg-0lax">iCaRL</td>
    <td class="tg-0lax">78.09/0.60</td>
    <td class="tg-0lax">6.09/0.23</td>
    <td class="tg-0lax">17.03/0.60</td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
  </tr>
  <tr>
    <td class="tg-0lax">GDumb</td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
  </tr>
  <tr>
    <td class="tg-0lax">BiC</td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
  </tr>
  <tr>
    <td class="tg-0lax"><span style="font-weight:bold">RM w/o DA</span></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
  </tr>
  <tr>
    <td class="tg-0lax"><span style="font-weight:bold">RM</span></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
  </tr>
</tbody>
</table>
 


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

