import requests
import base64
from gtts import gTTS
import io

SPECIFIC_DISEASE_ADVISORIES = {
    "Tomato_Leaf_Mold": {
        "en": "Apply Copper Fungicide or Difenoconazole. Ensure proper greenhouse ventilation and avoid overhead irrigation to lower leaf wetness.",
        "ta": "தக்காளி இலை காளானை கட்டுப்படுத்த காப்பர் பூஞ்சாணக்கொல்லியை தெளிக்கவும். காற்றோட்டத்தை அதிகரிக்கவும்.",
        "hi": "टमाटर के पत्ते के मोल्ड के लिए कॉपर फफूंदनाशी का छिड़काव करें। खेत में वेंटिलेशन में सुधार करें।",
        "te": "టమోటా ఆకు అచ్చు నివారణకు రాగి శిలీంధ్ర నాశిని పిచிகారీ చేయండి."
    },
    "Potato___Early_blight": {
        "en": "Spray Mancozeb or Chlorothalonil every 7-10 days. Maintain balanced nitrogen fertilization.",
        "ta": "உருளைக்கிழங்கு ஆரம்ப பிளைட் நோய்க்கு மான்கோசெப் தெளிக்கவும்.",
        "hi": "आलू के शुरुआती झुलसा रोग के लिए मैनकोजेब का प्रयोग करें।",
        "te": "பங்காళாதுంప తొలి తెగులు నివారణకు మాంకోజెబ్ పిచிகారీ చేయండి."
    }
}

class RAGEngine:
    def __init__(self):
        print("[SUCCESS] RAG Engine Initialized in Knowledge Base Mode!")

    def fetch_weather(self, lat: float, lon: float) -> dict:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true&hourly=relative_humidity_2m"
        try:
            response = requests.get(url, timeout=3)
            if response.status_code == 200:
                data = response.json()
                curr = data.get("current_weather", {})
                hourly = data.get("hourly", {})
                humi = hourly.get("relative_humidity_2m", [70]*24)[:24]
                risk = "High Risk" if max(humi) > 80 else "Moderate Risk"
                return {
                    "temperature": curr.get("temperature", 25.0),
                    "windspeed": curr.get("windspeed", 10.0),
                    "risk_score": risk
                }
        except Exception as e:
            print(f"[WEATHER API EXCEPTION] {e}")
            
        return {"temperature": 25.0, "windspeed": 10.0, "risk_score": "Moderate Risk"}

    def generate_advisory(self, disease_name: str, lat: float, lon: float, lang: str = "en") -> dict:
        weather_info = self.fetch_weather(lat, lon)
        
        advisory_text = SPECIFIC_DISEASE_ADVISORIES.get(disease_name, {}).get(
            lang, f"Standard management for {disease_name}. Apply approved protective fungicides and ensure proper soil drainage."
        )

        audio_b64 = ""
        try:
            tts = gTTS(text=advisory_text, lang=lang if lang in ['en', 'ta', 'hi', 'te'] else 'en')
            fp = io.BytesIO()
            tts.write_to_fp(fp)
            fp.seek(0)
            audio_b64 = base64.b64encode(fp.read()).decode('utf-8')
        except Exception as e:
            print(f"[TTS ERROR] {e}")

        return {
            "disease_detected": disease_name,
            "weather_context": weather_info,
            "llm_advisory": advisory_text,
            "audio_b64": audio_b64
        }

rag_engine = RAGEngine()