import google.generativeai as palm
import os

# Configure the client library by providing your API key.
palm.configure(api_key=os.getenv("PALM_API_KEY"))

# Call the model and print the response.
def get_response(text):
    response = palm.chat(messages=text)
    print(response.last)
    return response.last