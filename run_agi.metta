! (import! &self babyagi)

 ;; Run the face recognition agent
! (face_recognition "./agents/Lebron_James.jpg")

 ;; Fetch the matched person's data from Neo4j using the recognized face
! (fetch_person_data "MatchedPersonID")

 ;; Process the fetched data with the together_ai agent
! (together_ai_process "PersonData")

 ;; Add the processed data to the running space knowledge
! (add-reduct &self (together_ai_process "PersonData"))

 ;; Retrieve and display all the knowledge present in the running space
! (match &self $x $x)
