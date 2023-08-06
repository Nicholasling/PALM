import google.generativeai as palm
import base64
import json

# Configure the client library by providing your API key.
palm.configure(api_key="")

# These parameters for the model call can be set by URL parameters.
model = 'models/text-bison-001' # @param {isTemplate: true}
temperature = 0.7 # @param {isTemplate: true}
candidate_count = 1 # @param {isTemplate: true}
top_k = 40 # @param {isTemplate: true}
top_p = 0.95 # @param {isTemplate: true}
max_output_tokens = 1024 # @param {isTemplate: true}
text_b64 = 'd3JpdGUgYSBibG9nIHBvc3Qgb24gcmVuZXdhYmxlIGVuZXJneS4gTGltaXQgdGhlIG51bWJlciBvZiB3b3JkcyB0byAyMDAuIEZvcm1hdCB0aGUgb3V0cHV0IGluIEhUTUwu' # @param {isTemplate: true}
stop_sequences_b64 = 'W10=' # @param {isTemplate: true}
safety_settings_b64 = 'W3siY2F0ZWdvcnkiOiJIQVJNX0NBVEVHT1JZX0RFUk9HQVRPUlkiLCJ0aHJlc2hvbGQiOjF9LHsiY2F0ZWdvcnkiOiJIQVJNX0NBVEVHT1JZX1RPWElDSVRZIiwidGhyZXNob2xkIjoxfSx7ImNhdGVnb3J5IjoiSEFSTV9DQVRFR09SWV9WSU9MRU5DRSIsInRocmVzaG9sZCI6Mn0seyJjYXRlZ29yeSI6IkhBUk1fQ0FURUdPUllfU0VYVUFMIiwidGhyZXNob2xkIjoyfSx7ImNhdGVnb3J5IjoiSEFSTV9DQVRFR09SWV9NRURJQ0FMIiwidGhyZXNob2xkIjoyfSx7ImNhdGVnb3J5IjoiSEFSTV9DQVRFR09SWV9EQU5HRVJPVVMiLCJ0aHJlc2hvbGQiOjJ9XQ==' # @param {isTemplate: true}

# Convert the prompt text param from a bae64 string to a string.
text = base64.b64decode(text_b64).decode("utf-8")

# Convert the stop_sequences and safety_settings params from base64 strings to lists.
stop_sequences = json.loads(base64.b64decode(stop_sequences_b64).decode("utf-8"))
safety_settings = json.loads(base64.b64decode(safety_settings_b64).decode("utf-8"))

defaults = {
  'model': model,
  'temperature': temperature,
  'candidate_count': candidate_count,
  'top_k': top_k,
  'top_p': top_p,
  'max_output_tokens': max_output_tokens,
  'stop_sequences': stop_sequences,
  'safety_settings': safety_settings,
}

# Call the model and print the response.
def get_response(text):
    response = palm.generate_text(
        **defaults,
        prompt=text
    )
    print(response.result)
    return response.result
