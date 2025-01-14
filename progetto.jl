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
X_coerced = coerce(data, :TIPOLOGIA => Multiclass, Symbol("CODICE COL") => Multiclass);
imputer = FillImputer()
mach = machine(imputer, X_coerced) |> fit!
X_imputed = MLJ.transform(mach, X_coerced);


encoder = ContinuousEncoder()
mach = machine(encoder, X_imputed) |> fit!
X_encoded = MLJ.transform(mach, X_imputed)
file_path = "dataset-incendi-vito-martella-preprocessed.csv"
CSV.write(file_path, X_encoded)

#y, X = unpack(X_encoded, ==([Symbol("CODICE COL__Arancione"), Symbol("CODICE COL__Bianco"), Symbol("CODICE_COL__Giallo"), Symbol("CODICE_COL__Rosso"), Symbol("CODICE_COL__Verde")]), rng=778085);

# Definisci le colonne da separare
cols_to_unpack = [
    Symbol("CODICE COL__Arancione"),
    Symbol("CODICE COL__Bianco"),
    Symbol("CODICE COL__Giallo"),
    Symbol("CODICE COL__Rosso"),
    Symbol("CODICE COL__Verde")
]

# Estrai le colonne specificate in y
y_1= select(X_encoded, cols_to_unpack)

# Estrai le altre colonne in X
X_1 = select(X_encoded, Not(cols_to_unpack))
y, X = unpack(X_imputed, ==(Symbol("CODICE COL")), rng=778085)

SM= models(matching(X, y)) # supervised model
NNmodels = models(matching(X_1, y_1))# neural netwprd

## 
#Pkg.add("SymbolicRegression")
MultitargetSRRegressor = @load MultitargetSRRegressor pkg = SymbolicRegression
model1 = MultitargetSRRegressor()

split_index = Int(round(0.8 * nrow(X_1)))
trainX_1 = first(X_1, split_index)
testX_1 = last(X_1, nrow(X_1) - split_index)

trainY_1 = first(y_1, split_index)
testY_1 = last(y_1, nrow(y_1) - split_index)

MSRRloaded = machine(model1, trainX_1, trainY_1)

fit!(MSRRloaded)

r = report(MSRRloaded)
for (output_index, (eq, i)) in enumerate(zip(r.equation_strings, r.best_idx))
    println("Equation used for ", output_index, ": ", eq[i])
end