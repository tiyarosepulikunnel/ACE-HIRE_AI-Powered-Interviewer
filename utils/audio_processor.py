def get_indian_languages():
    """Returns a list of all 22 officially recognized Indian languages"""
    return [
        "English",
        "Hindi",
        "Bengali",
        "Telugu",
        "Marathi",
        "Tamil",
        "Urdu",
        "Gujarati",
        "Kannada",
        "Odia",
        "Malayalam",
        "Punjabi",
        "Assamese",
        "Maithili",
        "Sanskrit",
        "Sindhi",
        "Kashmiri",
        "Konkani",
        "Manipuri (Meitei)",
        "Nepali",
        "Bodo",
        "Dogri",
        "Santali"
    ]

def get_language_code(language_name):
    """Convert language name to ISO code for API"""
    language_map = {
        "English": "en",
        "Hindi": "hi",
        "Bengali": "bn",
        "Telugu": "te",
        "Marathi": "mr",
        "Tamil": "ta",
        "Urdu": "ur",
        "Gujarati": "gu",
        "Kannada": "kn",
        "Odia": "or",
        "Malayalam": "ml",
        "Punjabi": "pa",
        "Assamese": "as",
        "Maithili": "mai",
        "Sanskrit": "sa",
        "Sindhi": "sd",
        "Kashmiri": "ks",
        "Konkani": "kok",
        "Manipuri (Meitei)": "mni",
        "Nepali": "ne",
        "Bodo": "brx",
        "Dogri": "doi",
        "Santali": "sat"
    }
    
    return language_map.get(language_name, "en")