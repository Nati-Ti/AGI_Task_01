from neo4j import GraphDatabase
from dotenv import load_dotenv
import os

# Initialize Neo4j driver
def load_data(uri, user, password, cypher_file_path):
    driver = GraphDatabase.driver(uri, auth=(user, password))
    
    # Read the Cypher script
    with open(cypher_file_path, 'r') as file:
        cypher_script = file.read()
    
    with driver.session() as session:
        session.run(cypher_script)
        print("Data loaded into Neo4j.")

URL = os.getenv('NEO4j_URL')
USER = os.getenv('NEO4j_USER')
PASSWORD = os.getenv('NEO4j_PASSWORD')
CYPHER_FILE_PATH = './01_data.cypher'

load_data(URL, USER, PASSWORD, CYPHER_FILE_PATH)


class Neo4jFetcher:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def fetch_person_data(self, person_id):
        query = """
        MATCH (p:Person {id: $person_id})-[:HAS_SKILL]->(s:Skill)
        RETURN p.name AS name, collect(s.name) AS skills
        """
        with self.driver.session() as session:
            result = session.run(query, person_id=person_id)
            record = result.single()
            if record:
                return {
                    "name": record["name"],
                    "skills": record["skills"]
                }
        return None

    # def close(self):
    #     self.driver.close()
