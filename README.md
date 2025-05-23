<div align="center">
  <h2><b> Code for Paper:</b></h2>
  <h2><b> 	
Bridging the Spectrum Gap: Mid‑Frequency Augmentation and Key‑Frequency Mining for Multivariate Time Series </b></h2>
</div>

## Get Started

1. Install requirements. ```pip install -r requirements.txt```
2. Download data. You can download the all datasets from iTransformer: [datasets](https://drive.google.com/u/0/uc?id=1NF7VEefXCmXuWNbnNe858WvQAkJ_7wuP&export=download). **All the datasets are well pre-processed** and can be used easily.
3. Train the model. We provide the experiment scripts of all benchmarks under the folder `./scripts`. 
4. You can use bash commands to individually run scripts in the 'scripts' folder from the command line to obtain results for individual datasets, for example, you can use the below command line to obtain the multivariate forecasting result of  **input-96-predict-96** on ETTh1:
   
      ```bash ./scripts/96/ETTh1.sh ```

You can find:

The training history and results under './logs' folder.

The test results (MSE and MAE) under './test_dicts' folder.

The visualization of forecasting results under './visual' folder.

The visualization of the weight under './visual/weight' folder.  **PS: The method used for getting the weight follows the approach outlined in [An Analysis of Linear Time Series Forecasting Models](https://arxiv.org/abs/2403.14587).**

## Several examples for better understanding the proposed Energy-based Key-Frequency Picking Block(EKPB) in the paper.

Within the storage limits of a GitHub repository, We provide training log files of tasks **input-96-predict-96** on **ECL**, to assist readers in reproducing the results from the paper. 
