import os
import time
import google.generativeai as genai

from dotenv import load_dotenv

load_dotenv()

genai.configure(
    api_key=os.getenv("GEMINI_API_KEY")
)

model = genai.GenerativeModel(
    "gemini-2.5-flash"
)

# ==========================================
# SHORT FALLBACK ADVICE
# ==========================================

def fallback_advice(
    crop_name,
    crop_confidence,
    disease_name,
    disease_confidence
):

    return f"""
🌱 Recommended Crop
{crop_name} ({crop_confidence:.2f}%)

🦠 Disease
{disease_name} ({disease_confidence:.2f}%)

📌 Why Suitable
The soil and environmental conditions are favorable for cultivating {crop_name}.

✅ Recommended Actions
• Apply balanced fertilizers
• Maintain proper irrigation
• Monitor crop health regularly
• Remove infected leaves

⚠ Precautions
• Avoid excess moisture
• Use certified seeds
• Inspect crops weekly

🎯 Expected Outcome
Higher yield and healthier crops.
"""


# ==========================================
# GEMINI ADVISOR
# ==========================================

def generate_advisory(
    crop_name,
    crop_confidence,
    disease_name,
    disease_confidence
):

    prompt = f"""
You are an agricultural expert.

Crop:
{crop_name}

Crop Confidence:
{crop_confidence:.2f}%

Disease:
{disease_name}

Disease Confidence:
{disease_confidence:.2f}%

Generate a SHORT advisory.

Format:

🌱 Recommended Crop

🦠 Disease

📌 Why Suitable
(Max 2 lines)

✅ Recommended Actions
(Max 4 bullet points)

⚠ Precautions
(Max 3 bullet points)

🎯 Expected Outcome
(1 line)

Keep the entire response under 120 words.
Use simple farmer-friendly language.
"""

    try:

        for attempt in range(3):

            try:

                response = model.generate_content(
                    prompt
                )

                return response.text

            except Exception:

                print(
                    f"Retrying Gemini... Attempt {attempt+1}"
                )

                time.sleep(10)

        raise Exception("Gemini unavailable")

    except Exception as e:

        print("\nGemini Error:")
        print(e)

        print(
            "\nUsing fallback advisor...\n"
        )

        return fallback_advice(
            crop_name,
            crop_confidence,
            disease_name,
            disease_confidence
        )


# ==========================================
# TEST
# ==========================================

if __name__ == "__main__":

    report = generate_advisory(
        crop_name="Rice",
        crop_confidence=94.33,
        disease_name="Brown Spot",
        disease_confidence=97.10
    )

    print("\n")
    print("=" * 60)
    print("SMART AGRICULTURAL ADVISORY")
    print("=" * 60)
    print(report)