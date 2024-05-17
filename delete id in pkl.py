import pickle


def delete_id_and_encodings(known_face_encodings, id_to_delete):
    """
    Delete the specified ID and its corresponding encodings from the dictionary.

    Args:
        known_face_encodings (dict): Dictionary containing face encodings.
        id_to_delete (int): ID to be deleted.

    Returns:
        dict: Updated dictionary after deleting the specified ID and its encodings.
    """
    if id_to_delete in known_face_encodings:
        del known_face_encodings[id_to_delete]
        print(f"ID {id_to_delete} and its encodings have been deleted.")
    else:
        print(f"ID {id_to_delete} not found in the dictionary.")

    return known_face_encodings


if __name__ == "__main__":
    # Load the known_face_encodings dictionary from the .pkl file
    with open('known_face_encodings.pkl', 'rb') as f:
        known_face_encodings = pickle.load(f)

    # Get the ID to delete from user input
    id_to_delete = input("Enter the ID to delete:")

    # Delete the specified ID and its encodings
    updated_encodings = delete_id_and_encodings(known_face_encodings, id_to_delete)

    # Save the updated dictionary back to the .pkl file
    with open('known_face_encodings.pkl', 'wb') as f:
        pickle.dump(updated_encodings, f)


