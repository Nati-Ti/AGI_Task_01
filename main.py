from hyperon import OperationAtom, AtomType
from hyperon.ext import register_atoms
from face_recognition_agent import FaceRecognitionAgent
from together_ai_agent import TogetherAIProcessor
from pineconeDB import PineconeManager
from neoDB import Neo4jFetcher
from dotenv import load_dotenv
import os

URL = os.getenv('NEO4j_URL')
USER = os.getenv('NEO4j_USER')
PASSWORD = os.getenv('NEO4j_PASSWORD')
TOGETHERKEY = os.getenv('TOGETHERAIKEY')
PINCONEKEY = os.getenv('PINECONEAPIKEY')

face_recognition_agent = FaceRecognitionAgent()
neo4j_fetcher = Neo4jFetcher(uri=URL, user=USER, password=PASSWORD)
pinecone_manager = PineconeManager(api_key=PINCONEKEY, index_name="face-encodings")
together_ai_processor = TogetherAIProcessor(api_key=TOGETHERKEY)

# Registering agents
@register_atoms(pass_metta=True)
def babyagi_atoms(metta):
    together_ai_atom = OperationAtom('together_ai', lambda *args: together_ai_processor.process_with_together_ai(metta, *args), [AtomType.ATOM, "Expression"], unwrap=False)
    face_recognition_atom = OperationAtom('face_recognition', lambda *args: face_recognition_agent.get_face_encoding(*args), unwrap=False)

    return {
        r"together_ai": together_ai_atom,
        r"face_recognition": face_recognition_atom,
    }

def process_image(image_path):
    # Step 1: Process the image to get the face encoding using the face recognition agent
    face_encoding = face_recognition_agent.get_face_encoding(image_path)
    
    if not face_encoding:
        return "No face detected in the image."

    # Step 2: Find a matching face encoding in Pinecone
    matching_id = pinecone_manager.find_matching_face(face_encoding)
    
    if matching_id:
        # Step 3: Fetch the corresponding data from Neo4j
        person_data = neo4j_fetcher.fetch_person_data(matching_id)
        
        if person_data:
            # Step 4: Process the data with Together AI
            summary = together_ai_processor.process_with_together_ai(person_data)
            return summary
    
    return "No match found."


if __name__ == "__main__":
    image_path = "/agents/Lebron_James.jpg"
    result = process_image(image_path)
    print(result)
