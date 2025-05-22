def get_user_by_email(email):
    # Fetch user from the database (this is a placeholder, implement your own logic)
    # user = fetch_from_database(email)
    user = None  # Replace with actual database call
    return user

def add_user(firstname, lastname, address, email, password):
    # Check if the user already exists
    existing_user = get_user_by_email(email)
    if existing_user:
        return {"error": "User already exists"}

    # Create a new user
    new_user = {
        "firstname": firstname,
        "lastname": lastname,
        "address": address,
        "email": email,
        "password": password
    }
    
    # Save the new user to the database (this is a placeholder, implement your own logic)
    # save_to_database(new_user)
    
    return {"message": "User created successfully", "user": new_user}