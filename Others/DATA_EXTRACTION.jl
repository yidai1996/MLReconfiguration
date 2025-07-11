# Extracting data from csv file as input
using CSV, DataFrames
dir_initalconditions1 = "G:\\My Drive\\Research\\SVM\\Training dataset\\Raw data\\Initial conditions permutation\\parallel"
x1 = readdir(dir_initalconditions1)
for i in 1:length(x1)
    df = CSV.read("G:\\My Drive\\Research\\SVM\\Training dataset\\Raw data\\Setpoint tracking\\parallel\\"*x1[i],DataFrame)
    # operations of df
end
