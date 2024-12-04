import google.generativeai as genai

def configure_genai(api_key):
    if not api_key:
        raise ValueError("API key cannot be empty")
    try:
        genai.configure(api_key=api_key)
    except Exception as e:
        raise Exception(f"Failed to configure GEMINI AI: {str(e)}")

def get_health_recommendations(prediction, sex, age, accuracy):
    input_data = f"""
    target:{prediction}
    sex:{sex}
    age:{age}
    machine_learning_model_accuracy:{accuracy}
    """
    instruction = """
    Provide ways to reduce heart disease risk, recommend tests or doctor consultation if necessary,
    and give patient-friendly advice.
    """
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(instruction + input_data)
    return response.text
