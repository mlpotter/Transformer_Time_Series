# Transformer_Time_Series
DISLCLAIMER: THIS IS NOT THE PAPERS CODE. THIS DOES NOT HAVE SPARSITY. THIS IS TEACHER FORCED LEARNING. Only tried to replicate the simple example without sparsity.
[Enhancing the Locality and Breaking the Memory Bottleneck of Transformer on Time Series Forecasting](https://arxiv.org/pdf/1907.00235.pdf) (NeurIPS 2019)

Able to match the results of the paper for the synthetic dataset as shown in the table below
![Rp](https://github.com/mlpotter/Transformer_Time_Series/blob/master/images/Rp_table.JPG)

The synthetic dataset was constructed as shown below
![Synthetic Dataset](https://github.com/mlpotter/Transformer_Time_Series/blob/master/images/synthetic_datasets.JPG)

A nice visualization of how the attention layers look at the signal for predicting the last timestep t=t0+24-1
![Attention Visualization](https://github.com/mlpotter/Transformer_Time_Series/blob/master/images/attention.JPG)


![Learning Values (MSE)](https://github.com/mlpotter/Transformer_Time_Series/blob/master/images/learning_values.JPG)
![Learning Curve](https://github.com/mlpotter/Transformer_Time_Series/blob/master/images/learning_curve.JPG)
![Validation Example](https://github.com/mlpotter/Transformer_Time_Series/blob/master/images/validation_example.JPG)
