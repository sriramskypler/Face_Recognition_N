import pickle


def load_face_encodings(file_path):
    try:
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        return {}


def save_face_encodings(file_path, data):
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)


# Load the existing known and unknown face encodings
known_face_encodings = load_face_encodings('known_face_encodings.pkl')

# Specify the unknown ID to be converted and the new name
unknown_id_to_convert = input("Enter the ID to update: ")

new_known_name = input("Enter the name to update: ")


# Transfer the encodings from unknown to known
if unknown_id_to_convert in known_face_encodings:
    # Add the face encodings to the known faces under the new name
    known_face_encodings[new_known_name] = known_face_encodings[unknown_id_to_convert]

    del known_face_encodings[unknown_id_to_convert]

    # Save the updated data back to the .pkl files
    save_face_encodings('known_face_encodings.pkl', known_face_encodings)

    print(f"Successfully updated '{unknown_id_to_convert}' to '{new_known_name}'")
else:
    print(f"No face encodings found for ID '{unknown_id_to_convert}'")

#completed