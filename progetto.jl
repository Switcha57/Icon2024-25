using Pkg

Pkg.activate("my_MLJ_env")
# Pkg.add("MLJ")
# Pkg.add("CSV")
# Pkg.add("DataFrames")

using CSV
using DataFrames

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