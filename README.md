# SOD-NAS
Diabetic retinopathy, a retinal disease stemming from diabetes, is a leading global cause of blindness. Early detection of diabetic retinopathy is crucial for saving patients' lives and preserving their vision. The potential for early diabetic retinopathy diagnosis has been demonstrated through the utilization of deep learning technologies. However, several limitations persist, including the inflexibility of manual network design, the inefficiency of neural architecture search (NAS), and the information loss during the search process. To address these issues, we propose a second-order optimization differential architecture search (SOD-NAS) algorithm for predicting diabetic retinopathy. Firstly, we utilize neural architecture search algorithms for grading diabetic retinopathy to overcome the limitations of manual network design. Then, we propose second-order optimization to address the slow convergence and high error issues associated with traditional NAS. Finally, we integrate the Gumbel-Softmax sampling method into NAS to encompass all operation information during normalization. For a detailed description of technical details and experimental results, please refer to our paper:

SOD-NAS: Diabetic Retinopathy Prediction Algorithm Based on Second-Order Optimization and Differential Architecture Search

## DATASETS

IDRiD, MESSIDOR, EYEPACS, DDR require manual downloading, with corresponding paper [IDRiD](https://doi.org/10.3390/data3030025), [MESSIDOR](https://doi.org/10.5566/IAS.1155), [EYEPACS](https://doi.org/10.1177/193229680900300315), [DDR](https://doi.org/10.1016/j.ins.2019.06.011).

## Prerequisite for server

```
Python = 3.8, PyTorch == 2.0.1, torchvision == 0.15.2
```

## Architecture search

To carry out architecture search using 2nd-order, run (e.g. IDRiD dataset)

```
python train_IDRiD.py --unrolled
```

## Architecture evaluation

To evaluate our best cells by training from scratch, run (e.g. IDRiD dataset)

```
python train-IDRiD-evaluation.py --auxiliary --cutout
```

Customized architectures are supported through the ***--arch*** flag once specified in ***genotypes.py***.

## Visualization

Package [graphviz](https://graphviz.readthedocs.io/en/stable/index.html) is required to visualize the learned cells

```
python visualize.py SOD-NAS
```

where ***SOD-NAS*** can be replaced by any customized architectures in ***genotypes.py***.
