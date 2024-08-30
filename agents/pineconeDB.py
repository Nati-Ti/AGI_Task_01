import pinecone

class PineconeManager:
    def __init__(self, api_key, index_name):
        pinecone.init(api_key=api_key)
        self.index = pinecone.Index(index_name)

    def store_face_encoding(self, id, face_encoding):
        self.index.upsert(vectors=[(id, face_encoding)])

    def find_matching_face(self, face_encoding):
        result = self.index.query(queries=[face_encoding], top_k=1, include_values=True)
        if result['matches']:
            best_match = result['matches'][0]
            similarity_score = best_match['score']
            matching_id = best_match['id']
            if similarity_score > 0.7:  
                return matching_id
        return None
