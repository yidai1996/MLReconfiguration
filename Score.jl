using CSV, DataFrames

function String_to_Number(a)
    println(a)
    c = 0 
    if a == "parallel"
        c = 1.0
    elseif a == "hybrid"
        c = 2.0
    elseif a == "mixing"
        c = 3.0
    elseif a == "series"
        c = 4.0
    else 
        c = 5.0
    end
end


df = CSV.read("C:\\Users\\yid\\TemporaryResearchDataStorage\\Reconfiguration\\dataset\\KNN_Zavreal_kernel_95.52_sorted.csv",DataFrame,types=Dict(1=>Float64))
Predict = df.PredictedBestConfiguration
Best = df.BestConfiguration
row_number = nrow(df)
predict_index = zeros(row_number);
best_index = zeros(row_number);
Predict_pi = zeros(row_number);
Best_pi = zeros(row_number);
df_score = df[:,[:parallel, :hybrid, :mixing, :series]]
pi_allconfiguration = Matrix(df_score)
sorted_configuration = Matrix(df[:,[:BestConfiguration, :SecondBestConfiguration, :ThirdBestConfiguration, :WorstBestConfiguration]])

# If need to know how best the configuration is (best, second best, third best, worst), use the following code
# sorted_configuration = Matrix(df[:,[:BestConfiguration, :SecondBestConfiguration, :ThirdBestConfiguration, :WorstBestConfiguration]])
# for i in 1:row_number
#     position = findall(x-> x == Predict[i], sorted_configuration[i,:])
#     println(position)
# end

for i in 1:row_number
    position = findall(x-> x == Predict[i], sorted_configuration[i,:])
    a = sorted_configuration[i,position]
    println(i)
    b = a[1]
    predict_index[i] = String_to_Number(b)
    if predict_index[i] == 5.0
        println("ERROR!")
        break
    end
    best_index[i] = String_to_Number(Best[i])
    
    Predict_pi[i] = pi_allconfiguration[i, convert(Int, predict_index[i])]
    Best_pi[i] = pi_allconfiguration[i, convert(Int, best_index[i])]
end
position = findall(x-> x == Predict[3], sorted_configuration[3,:])
a = sorted_configuration[3,position]
String_to_Number(a)
println(predict_index[3])

for i in 1:row_number
    println(Predict_pi[i], " ", Best_pi[i])
end

bbb = exp.((- Predict_pi + Best_pi)./Best_pi)

score1 = sum(bbb)/row_number

index_smallscore = findall(x -> x < 0.95, bbb)
present_smallcase_score = bbb[index_smallscore]
present_smallcase_pi = Predict_pi[index_smallscore]
present_smallcase_best_pi = Best_pi[index_smallscore]

# If need to know how best the configuration is (best, second best, third best, worst), use the following code
sorted_configuration2 = Matrix(df[:,[:BestConfiguration, :SecondBestConfiguration, :ThirdBestConfiguration, :WorstBestConfiguration]])
predict_score2 = zeros(row_number);
best_score2 = zeros(row_number);
for i in 1:row_number
    predict_score2[i] = findall(x-> x == Predict[i], sorted_configuration[i,:])[1]
    best_score2[i] = findall(x-> x == Best[i], sorted_configuration[i,:])[1]
end
element_comparsion = isequal.(predict_score2, best_score2)
different_cases = findall(x -> x == 0, element_comparsion)
different_predict = predict_score2[different_cases]
different_number = length(different_predict)
second_best_number = length(findall(x -> x == 2, different_predict))
third_best_number = length(findall(x -> x == 3, different_predict))
worst_number =  length(findall(x -> x == 4, different_predict))

(row_number - third_best_number - worst_number)/row_number