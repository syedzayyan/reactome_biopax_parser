from neo4j import GraphDatabase

uri = "bolt://localhost:7687"
username = "neo4j"
password = "ZayyanGraphDB123"  # whatever you set

driver = GraphDatabase.driver(uri, auth=(username, password))


def run_query(tx, query):
    result = tx.run(query)
    return [record for record in result]


with driver.session() as session:
    result = session.execute_read(run_query, "MATCH (n) RETURN count(n) AS count")
    print(result)

driver.close()
