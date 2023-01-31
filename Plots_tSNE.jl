# https://github.com/lejon/TSne.jl

using TSne, Statistics, MLDatasets, CSV, DataFrames, Plots

df = CSV.read("G:\\My Drive\\Research\\SVM\\Training dataset\\SVM results.csv",DataFrame)
count_row = nrow(df)
count_feature = ncol(df) - 2
# df1 = Matrix{2}(df)
df1 = Matrix(df)
A = df1[:,1:9]
TestLabel = df1[:,10]
PredictLabel = df1[:,11]

rescale(A; dims=1) = (A .- mean(A, dims=dims)) ./ max.(std(A, dims=dims), eps())

alldata, allabels = MNIST.traindata(Float64);
# alldata, allabels = MNIST(split=:train);
# data = reshape(permutedims(alldata[:, :, 1:2500], (3, 1, 2)),
            #    2500, size(alldata, 1)*size(alldata, 2));
# Normalize the data, this should be done if there are large scale differences in the dataset
X = rescale(A, dims=1);

Y = tsne(X, 2, 0, 1000, 20.0);

len = 1
Test_Y1_parallel = zeros(len);
Test_Y1_hybrid = zeros(len);
Test_Y1_mixing = zeros(len);
Test_Y1_series = zeros(len);
Test_Y2_parallel = zeros(len);
Test_Y2_hybrid = zeros(len);
Test_Y2_mixing = zeros(len);
Test_Y2_series = zeros(len);
Predict_Y1_parallel = zeros(len);
Predict_Y1_hybrid = zeros(len);
Predict_Y1_mixing = zeros(len);
Predict_Y1_series = zeros(len);
Predict_Y2_parallel = zeros(len);
Predict_Y2_hybrid = zeros(len);
Predict_Y2_mixing = zeros(len);
Predict_Y2_series = zeros(len);

count_total = 0
Test_count_p = 0
Test_count_h = 0
Test_count_m = 0
Test_count_s = 0
Predict_count_p = 0
Predict_count_h = 0
Predict_count_m = 0
Predict_count_s = 0

for i in 1:size(Y)[1]
    count_total += 1
    # Go through all test data
    if TestLabel[i] == "parallel"
        # println("Find parallel")
        Test_count_p += 1
        if Test_count_p == 1
            Test_Y1_parallel[Test_count_p] = Y[i,1]
            Test_Y2_parallel[Test_count_p] = Y[i,2]
        else 
            append!(Test_Y1_parallel, Y[i,1])
            append!(Test_Y2_parallel, Y[i,2])
            # println(Test_Y2_parallel)
        end

    elseif TestLabel[i] == "hybrid"
        Test_count_h += 1
        if Test_count_h == 1
            Test_Y1_hybrid[Test_count_h] = Y[i,1]
            Test_Y2_hybrid[Test_count_h] = Y[i,2]
        else 
            append!(Test_Y1_hybrid, Y[i,1])
            append!(Test_Y2_hybrid, Y[i,2])
        end

    elseif TestLabel[i] == "mixing"
        Test_count_m += 1
        if Test_count_m == 1
            Test_Y1_mixing[Test_count_m] = Y[i,1]
            Test_Y2_mixing[Test_count_m] = Y[i,2]
        else 
            append!(Test_Y1_mixing, Y[i,1])
            append!(Test_Y2_mixing, Y[i,2])
        end
    
    else
        Test_count_s += 1
        if Test_count_s == 1
            Test_Y1_series[Test_count_s] = Y[i,1]
            Test_Y2_series[Test_count_s] = Y[i,2]
        else 
            append!(Test_Y1_series, Y[i,1])
            append!(Test_Y2_series, Y[i,2])
        end
    end


    # Go through all predict data
    if PredictLabel[i] == "parallel"
        # println("Find parallel")
        Predict_count_p += 1
        if Predict_count_p == 1
            Predict_Y1_parallel[Predict_count_p] = Y[i,1]
            Predict_Y2_parallel[Predict_count_p] = Y[i,2]
        else 
            append!(Predict_Y1_parallel, Y[i,1])
            append!(Predict_Y2_parallel, Y[i,2])
            # println(Predict_Y2_parallel)
        end

    elseif PredictLabel[i] == "hybrid"
        Predict_count_h += 1
        if Predict_count_h == 1
            Predict_Y1_hybrid[Predict_count_h] = Y[i,1]
            Predict_Y2_hybrid[Predict_count_h] = Y[i,2]
        else 
            append!(Predict_Y1_hybrid, Y[i,1])
            append!(Predict_Y2_hybrid, Y[i,2])
        end

    elseif PredictLabel[i] == "mixing"
        Predict_count_m += 1
        if Predict_count_m == 1
            Predict_Y1_mixing[Predict_count_m] = Y[i,1]
            Predict_Y2_mixing[Predict_count_m] = Y[i,2]
        else 
            append!(Predict_Y1_mixing, Y[i,1])
            append!(Predict_Y2_mixing, Y[i,2])
        end
    
    else
        Predict_count_s += 1
        if Predict_count_s == 1
            Predict_Y1_series[Predict_count_s] = Y[i,1]
            Predict_Y2_series[Predict_count_s] = Y[i,2]
        else 
            append!(Predict_Y1_series, Y[i,1])
            append!(Predict_Y2_series, Y[i,2])
        end
    end
    if count_total == size(Y)[1]
        println("Finish going through all rows")
        break
    end
end

scatter(Predict_Y2_parallel, Predict_Y1_parallel, ms = 5, markershape = :+, markercolor = :red, label = "Predicted_parallel")
scatter!(Predict_Y2_hybrid, Predict_Y1_hybrid, ms = 5, markershape = :+, markercolor = :green, label = "Predicted_hybrid")
scatter!(Predict_Y2_mixing, Predict_Y1_mixing, ms = 5, markershape = :+, markercolor = :blue, label = "Predicted_mixing")
scatter!(Predict_Y2_series, Predict_Y1_series, ms = 5, markershape = :+, markercolor = :black, label = "Predicted_series")

scatter!(Test_Y2_parallel, Test_Y1_parallel, ms = 2, markershape = :none, markercolor = :red, label = "Tested_parallel")
scatter!(Test_Y2_hybrid, Test_Y1_hybrid, ms = 2, markershape = :none, markercolor = :green, label = "Tested_hybrid")
scatter!(Test_Y2_mixing, Test_Y1_mixing, ms = 2, markershape = :none, markercolor = :blue, label = "Tested_mixing")
scatter!(Test_Y2_series, Test_Y1_series, ms = 2, markershape = :none, markercolor = :black, label = "Tested_series")


# ColorLabel_Test = String[]


# for i in 1:size(Y)[1]
#     # if TestLabel[i] == String15("parallel")
#     if TestLabel[i] == "parallel"
#         push!(ColorLabel_Test, "red")
#         # ColorLabel_Test[i] = 1
#         println("find parallel")
#     elseif TestLabel[i] == "hybrid"
#         push!(ColorLabel_Test, "blue")
#         # ColorLabel_Test[i] = "blue"
#         # ColorLabel_Test[i] = 4
#     elseif TestLabel[i] == "mixing"
#         push!(ColorLabel_Test, "gold")
#         # ColorLabel_Test[i] = "gold"
#         # ColorLabel_Test[i] = 6
#     else
#         push!(ColorLabel_Test, "limegreen")
#         # ColorLabel_Test[i] = "limegreen"
#         # ColorLabel_Test[i] = 0
#     end
#     # println(TestLabel[i])
# end

# # theplot = scatter(Y[:,1], Y[:,2], marker=(2,2), color=Int.(ColorLabel_Test[1:size(Y,1)]))

# theplot = scatter(Y[:,1], Y[:,2], marker=(2,2), color=:"red")
# # Plots.pdf(theplot, "myplot.pdf")