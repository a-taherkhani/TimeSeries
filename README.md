# TimeSeries
# A Deep Convolutional Neural Network for Time Series Classification with Intermediate Targets

For All use of the data, please 'cite' the following:

'Taherkhani, A., Cosma, G. & McGinnity, T.M. A Deep Convolutional Neural Network for Time Series Classification with Intermediate Targets. SN COMPUT. SCI. 4, 832 (2023). https://doi.org/10.1007/s42979-023-02159-4'

The paper is avilabel at: (https://link.springer.com/article/10.1007/s42979-023-02159-4)

# How to run
- Put your time series data in '/archives/UCR_TS_Archive_2015'. Note that we have put one sample data in that folder called/RandTS/ that can be used to run the 2 methods,ie Proposed 'CNN-TS' and 'Base DNN'
- To run the proposed 'CNN-TS' run ‘run3_o3.py’.
- When you run the method on a dataset, it saves the results in a folder called ‘Results’. Note that if you have already run the method on the same dataset and have its results in the ‘results’ folder, you cannot run it again on the same data until you delete the results in the ‘Results’ folder.
- To Run 'Base DNN' on the sample dataset method you need to run 'run3_o1.py'

# Requirment
The code was tested in an environment in Anaconda. A list of the installed packages in the environment where the code was run is available in requirements.txt
