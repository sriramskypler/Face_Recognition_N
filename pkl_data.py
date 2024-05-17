
# This one is for tuple and dict
import pickle

# Load the known_face_encodings dictionary from the .pkl file
with open('known_face_encodings.pkl', 'rb') as f:
    known_face_encodings = pickle.load(f)

# Print the data type of the loaded object
data_type = type(known_face_encodings).__name__
print("Data type of the loaded object:", data_type)

# Initialize a dictionary to count the number of encodings for each unique ID
id_encodings_count = {}

# Iterate over the face encodings and count the number of encodings for each ID
for face_id, encodings in known_face_encodings.items():
    id_encodings_count[face_id] = len(encodings)

# Print the total number of unique IDs
print("Total number of unique IDs:", len(id_encodings_count))

# Print the number of encodings for each unique ID and their corresponding encodings
print("\nNumber of encodings for each unique ID and their encodings:")
for face_id, count in id_encodings_count.items():
    print(f"ID: {face_id}, Number of Encodings: {count}")
    print("Encodings:")
    # for i, encoding in enumerate(encodings):
    #     print(f"  Encoding {i + 1}: {encoding}")