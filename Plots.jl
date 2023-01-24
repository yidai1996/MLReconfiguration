using CSV, DataFrames, Plots

df = CSV.read("G:\\My Drive\\Research\\SVM\\Training dataset\\SVM results.csv",DataFrame)
# println(csv_reader)
len_row = length(eachrow(df))
len = 1
Test_T_feeding_parallel = zeros(len);
Test_T_feeding_hybrid = zeros(len);
Test_T_feeding_mixing = zeros(len);
Test_T_feeding_series = zeros(len);
Test_xBset_parallel = zeros(len);
Test_xBset_hybrid = zeros(len);
Test_xBset_mixing = zeros(len);
Test_xBset_series = zeros(len);
Predict_T_feeding_parallel = zeros(len);
Predict_T_feeding_hybrid = zeros(len);
Predict_T_feeding_mixing = zeros(len);
Predict_T_feeding_series = zeros(len);
Predict_xBset_parallel = zeros(len);
Predict_xBset_hybrid = zeros(len);
Predict_xBset_mixing = zeros(len);
Predict_xBset_series = zeros(len);
# T1_initial = zeros(len)
# T2_initial = zeros(len)
# T3_initial = zeros(len)
# xB1_initial = zeros(len)
# xB2_initial = zeros(len)
# xB3_initial = zeros(len)
# xBt_initial = zeros(len)
count_total = 0
Test_count_p = 0
Test_count_h = 0
Test_count_m = 0
Test_count_s = 0
Predict_count_p = 0
Predict_count_h = 0
Predict_count_m = 0
Predict_count_s = 0
for row in eachrow(df)
    count_total += 1
    # Go through all test data
    if row.BestConfiguration == "parallel"
        # println("Find parallel")
        Test_count_p += 1
        if Test_count_p == 1
            Test_T_feeding_parallel[Test_count_p] = row.Tin
            Test_xBset_parallel[Test_count_p] = row.xBset
        else 
            append!(Test_T_feeding_parallel, row.Tin)
            append!(Test_xBset_parallel, row.xBset)
            # println(Test_xBset_parallel)
        end

    elseif row.BestConfiguration == "hybrid"
        Test_count_h += 1
        if Test_count_h == 1
            Test_T_feeding_hybrid[Test_count_h] = row.Tin
            Test_xBset_hybrid[Test_count_h] = row.xBset
        else 
            append!(Test_T_feeding_hybrid, row.Tin)
            append!(Test_xBset_hybrid, row.xBset)
        end

    elseif row.BestConfiguration == "mixing"
        Test_count_m += 1
        if Test_count_m == 1
            Test_T_feeding_mixing[Test_count_m] = row.Tin
            Test_xBset_mixing[Test_count_m] = row.xBset
        else 
            append!(Test_T_feeding_mixing, row.Tin)
            append!(Test_xBset_mixing, row.xBset)
        end
    
    else
        Test_count_s += 1
        if Test_count_s == 1
            Test_T_feeding_series[Test_count_s] = row.Tin
            Test_xBset_series[Test_count_s] = row.xBset
        else 
            append!(Test_T_feeding_series, row.Tin)
            append!(Test_xBset_series, row.xBset)
        end
    end


    # Go through all predict data
    if row.PredictedBestConfiguration == "parallel"
        Predict_count_p += 1
        if Predict_count_p == 1
            Predict_T_feeding_parallel[Predict_count_p] = row.Tin
            Predict_xBset_parallel[Predict_count_p] = row.xBset
        else 
            append!(Predict_T_feeding_parallel, row.Tin)
            append!(Predict_xBset_parallel, row.xBset)
        end

    elseif row.PredictedBestConfiguration == "hybrid"
        Predict_count_h += 1
        if Predict_count_h == 1
            Predict_T_feeding_hybrid[Predict_count_h] = row.Tin
            Predict_xBset_hybrid[Predict_count_h] = row.xBset
        else 
            append!(Predict_T_feeding_hybrid, row.Tin)
            append!(Predict_xBset_hybrid, row.xBset)
        end

    elseif row.PredictedBestConfiguration == "mixing"
        Predict_count_m += 1
        if Predict_count_m == 1
            Predict_T_feeding_mixing[Predict_count_m] = row.Tin
            Predict_xBset_mixing[Predict_count_m] = row.xBset
        else 
            append!(Predict_T_feeding_mixing, row.Tin)
            append!(Predict_xBset_mixing, row.xBset)
        end
    
    else
        Predict_count_s += 1
        if Predict_count_s == 1
            Predict_T_feeding_series[Predict_count_s] = row.Tin
            Predict_xBset_series[Predict_count_s] = row.xBset
        else 
            append!(Predict_T_feeding_series, row.Tin)
            append!(Predict_xBset_series, row.xBset)
        end
    end

    if count_total == len
        println("Finish going through all rows")
        break
    end
end

scatter(Predict_xBset_parallel, Predict_T_feeding_parallel, ms = 5, markershape = :+, markercolor = :red, label = "Predicted_parallel")
scatter(Predict_xBset_hybrid, Predict_T_feeding_hybrid, ms = 5, markershape = :+, markercolor = :green, label = "Predicted_hybrid")
scatter(Predict_xBset_parallel, Predict_T_feeding_parallel, ms = 5, markershape = :+, markercolor = :blue, label = "Predicted_mixing")
scatter(Predict_xBset_parallel, Predict_T_feeding_parallel, ms = 5, markershape = :+, markercolor = :black, label = "Predicted_series")

scatter!(Test_xBset_parallel, Test_T_feeding_parallel, ms = 5, markershape = :none, markercolor = :red, label = "Tested_parallel")
scatter!(Test_xBset_hybrid, Test_T_feeding_hybrid, ms = 5, markershape = :none, markercolor = :green, label = "Tested_hybrid")
scatter!(Test_xBset_parallel, Test_T_feeding_parallel, ms = 5, markershape = :none, markercolor = :blue, label = "Tested_mixing")
scatter!(Test_xBset_parallel, Test_T_feeding_parallel, ms = 5, markershape = :none, markercolor = :black, label = "Tested_series")

