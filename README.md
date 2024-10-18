# SODAS
Diabetic retinopathy, a retinal disease resulting from diabetes, has become the leading cause of blindness. Early detection of diabetic retinopathy is crucial for preserving vision and saving lives. Up to now, some pioneering methods have been proposed for diagnosing diabetic retinopathy and have achieved preliminary results. However, several limitations still persist, including inability in detecting diabetic retinopathy at various scales and low accuracy in large-scale real scene. To this end, we, in this paper, we propose Second-Order Optimization Differential Architecture Search (SODAS) for predicting diabetic retinopathy. Firstly, we utilize neural architecture search to grade diabetic retinopathy to overcome limitations of manually designed networks which neglects retinopathy at different scales. Then, we integrate Gumbel-Softmax sampling method into neural architecture search to encompass all operation information during normalization, which is desired to reduce gradient information loss and improve accuracy. Additionally, we design second-order optimization to mitigate slow convergence associated with classical neural architecture search, which has been proven to be quickly convergent with detailed analysis and experimental results. Extensive experiments conducted on benchmark datasets demonstrate that SODAS has achieved average improvements of 19.5%, 24.3%, 32.2%, 35.6%, and 26.5% on Accuracy, Cohen’s Kappa, ROC-AUC, IBA, and F1-score, respectively. For a detailed description of technical details and experimental results, please refer to our paper:

SODAS: Second-Order Optimization Differential Architecture Search for Diabetic Retinopathy Prediction

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
python visualize.py SODAS
```

where ***SODAS*** can be replaced by any customized architectures in ***genotypes.py***.
