using Pkg

Pkg.activate("my_MLJ_env")
# Pkg.add("MLJ")
# Pkg.add("CSV")
# Pkg.add("DataFrames")
# Pkg.add("Plots")
#Pkg.add("PyPlot")
using CSV
using DataFrames
using Plots
using Dates
import PyPlot;
using MLJ
# Preprocessing 

# Read the CSV file
file_path = "dataset-incendi-vito-martella.csv"
data = CSV.read(file_path, DataFrame)

println("\nControllando la presenza di righe mal formate...")
malformed_rows = filter(row -> length(row) != ncol(data), eachrow(data))
if isempty(malformed_rows)
    println("Nessuna riga malformata.")
else
    println("Malformed rows found:")
    println(malformed_rows)
end

# Check for duplicate rows
println("\nChecking for duplicate rows...")
duplicate_rows = findall(nonunique(data))
if isempty(duplicate_rows)
    println("No duplicate rows found.")
else
    println("Duplicate rows found:")
    println(duplicate_rows)
end

# primi grafici 
# Grafico 1: Numero di incendi per anno
incendi_per_anno = combine(groupby(data, :ANNO), nrow => :Incendi)
p1 = bar(incendi_per_anno.ANNO, incendi_per_anno.Incendi, xlabel="Anno", ylabel="Numero di Incendi", title="Numero di Incendi per Anno")

# Grafico 2: Numero di incendi per comune
incendi_per_comune = combine(groupby(data, :COMUNE), nrow => :Incendi)
p2 = bar(incendi_per_comune.COMUNE, incendi_per_comune.Incendi, xlabel="Comune", ylabel="Numero di Incendi", title="Numero di Incendi per Comune", xticks=:auto, rotation=45)

# Grafico 3: Distribuzione delle tipologie di incendi
tipologie_incendi = combine(groupby(data, :TIPOLOGIA), nrow => :Incendi)
p3 = pie(tipologie_incendi.TIPOLOGIA, tipologie_incendi.Incendi, title="Distribuzione delle Tipologie di Incendi")

# Grafico 4: Numero di incendi per mese
data.MESE = month.(data.DATA)
incendi_per_mese = combine(groupby(data, :MESE), nrow => :Incendi)
p4 = bar(incendi_per_mese.MESE, incendi_per_mese.Incendi, xlabel="Mese", label="Numero incendi")

# Grafico 3: Distribuzione delle gravità di incendi
#tipologie_incendi = combine(groupby(data,Symbol("CODICE COL")), nrow => :Incendi)
#p3 = pie(tipologie_incendi[!,1], tipologie_incendi.Incendi, title="Distribuzione delle gravità di Incendi")


# 
# codice_col_counts = combine(groupby(data, Symbol("CODICE COL")), nrow => :count)

# # Estrai i valori e le etichette
# labels = codice_col_counts[!,1]
# sizes = codice_col_counts.count

# # Definisci i colori per ogni codice colore
# color_map = Dict(
#     "Arancione" => "orange",
#     "Rosso" => "red",
#     "Giallo" => "yellow",
#     "Verde" => "green",
#     "Bianco" => "white"
# )

# colors = [color_map[label] for label in labels]

# # Crea il grafico a torta
# PyPlot.figure(figsize=(8, 8))
# PyPlot.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=140)
# PyPlot.title("Distribuzione dei Codici Colore")
# PyPlot.axis("equal")  # Assicura che il grafico a torta sia disegnato come un cerchio
# PyPlot.show()
# display(gcf())


#data[!, Symbol("CODICE COL")] = replace(data[!, Symbol("CODICE COL")], replace_map_codice_col)

#data[!, Symbol("CODICE COL")] = Int64.(replace(data[!, Symbol("CODICE COL")], "Arancione" => 4, "Bianco" => 1, "Rosso" => 5,"Giallo" => 3,"Verde" => 2))
select!(data, Not(:COMUNE))
select!(data, Not(:DATA))

select!(data, Not(:PROV))

select!(data, Not(Symbol("LOCALITA'")))
data[!, :TIPOLOGIA] = Int64.(replace(data[!, :TIPOLOGIA], "Bosco" => 0, "Canneto" => 1, "Macchia" => 2))

X_coerced = coerce(data, Symbol("CODICE COL") => Multiclass);
imputer = FillImputer()
mach = machine(imputer, X_coerced) |> fit!
X_imputed = MLJ.transform(mach, X_coerced);


y, X = unpack(X_imputed, ==(Symbol("CODICE COL")), rng=778085)

# supervised Learning  Decision Tree Classifier
#Pkg.add("MLJDecisionTreeInterface")
#Pkg.add("MLJModels")
#Pkg.add("BetaML")
model = (@load DecisionTreeClassifier pkg = DecisionTree)()
model_constant = (@load ConstantClassifier pkg = MLJModels)()

mach = machine(model, X, y) |> fit!

min_samples_leaf_lambda = range(model, :min_samples_leaf, lower=1, upper=3000, scale=:log10)
curve = MLJ.learning_curve(mach;
     range=min_samples_leaf_lambda,
    measure=log_loss)
mach_constant = machine(model_constant, X, y)

fit!(mach_constant)
y_pred_constant = predict(mach_constant, X)
log_loss_constant = LogLoss()(y_pred_constant,y)

# plot(curve.parameter_values, curve.measurements, label="Decision Tree", xlabel="min_samples_leaf", ylabel="Log Loss", title="Learning Curve")

# plot!([1, 3000], [log_loss_constant, log_loss_constant], label="Constant Classifier", linestyle=:dash)


 modelNNC = (@load NeuralNetworkClassifier pkg = "BetaML" verbosity = 0)()
 mach_nn = machine(modelNNC, X, y)
# max_epoch_lambda = range(modelNNC, :epochs, lower=10, upper=300, scale=:log10)
fit!(mach_nn)
# curve_nn = MLJ.learning_curve(mach_nn,
# range=max_epoch_lambda,
#     measure=log_loss)
# plot(curve_nn.parameter_values, curve_nn.measurements, label="Neural Network Classifier Fitted", xlabel="epochs", ylabel="Log Loss", title="Learning Curve")
# plot!([1, 300], [log_loss_constant, log_loss_constant], label="Constant Classifier", linestyle=:dash)
# modelType =  @load RandomForestClassifier pkg = DecisionTree
# model_RFC = modelType()
# machRFC = machine(model_RFC, X,y);
# fit!(machRFC);

# max_depth_lambda = range(model, :min_samples_leaf, lower=1, upper=50, scale=:log2)
# curve_RFC = MLJ.learning_curve(machRFC;
#       range=max_depth_lambda,
#      measure=log_loss)
# plot(curve_RFC.parameter_values, curve_RFC.measurements, label="Random Forest Classifier", xlabel="min_samples_leaf", ylabel="Log Loss", title="Learning Curve")
# plot!([1, 50], [log_loss_constant, log_loss_constant], label="Constant Classifier", linestyle=:dash)
