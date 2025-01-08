from io import BytesIO
from PIL import Image
import base64
import google.generativeai as genai

# Configure the API key and model
genai.configure(api_key="AIzaSyC6FhO2BCsCoLp9aYjMnq59mzZ-hkKImi0")
model = genai.GenerativeModel(model_name="gemini-1.5-flash")


# Read and encode the image to a base64 string
def get_response(image_data):
    image_bytes = BytesIO(image_data)

    image = Image.open(image_bytes)

    # buffered = BytesIO()
    # image.save(buffered, format=image.format)
    # b64_string = base64.b64encode(buffered.getvalue()).decode('utf-8')
    

    # Define the prompt focused on image analysis
    prompt = (
        "You are a helpful and knowledgeable study buddy. You will analyze an image provided by the user "
        "and generate insights or answer any related questions as comprehensively as possible. "
        "Keep your explanations simple and clear, as if the user has no prior background knowledge. "
        "Assume no tools are available other than your interpretation of the image alone."
        "\n\n"
        "Respond with well-structured information, using backticks or LaTeX if you need to represent any mathematical symbols. "
        "Be friendly and supportive, making your answers engaging and focused on the user’s understanding."
        "If the prompt is unclear or ambiguous, politely ask for clarification to better assist the user."
    )

    # Generate the response
    response = model.generate_content([prompt, image])

    return response.text
