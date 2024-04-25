import sparql

# Define the endpoint URL of your GraphDB TCP server
endpoint = "tcp://localhost:7200"  # Replace with your GraphDB TCP endpoint


# Define your SPARQL query
query = ('SELECT ?subject ?predicate ?object WHERE { '
    '?subject ?predicate ?object } LIMIT 10')

# Execute the query and print the results
results = sparql.query(endpoint, query)

for row in results:
    print(row)
