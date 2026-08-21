from flask import Flask, request, jsonify, send_file
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import os

app = Flask(__name__)

MODEL_NAME = os.environ.get("MUSICGEN_MODEL", "small")
print(f"Loading MusicGen: facebook/musicgen-{MODEL_NAME}")
model = MusicGen.get_pretrained(MODEL_NAME)
print("Model loaded!")

@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json() or {}
    prompt = data.get("prompt", "")
    duration = min(int(data.get("duration", 30)), 300)
    
    if not prompt:
        return jsonify({"error": "prompt required"}), 400
    
    print(f"Generating: {prompt} ({duration}s)")
    model.set_generation_params(duration=duration)
    wav = model.generate([prompt])
    
    output_path = os.path.join(os.path.dirname(__file__), "temp_output.wav")
    audio_write(
        output_path.replace(".wav", ""),
        wav[0].cpu(),
        model.sample_rate,
        strategy="loudness",
        loudness_compressor=True
    )
    
    return send_file(output_path, mimetype="audio/wav",
                     as_attachment=True,
                     download_name=f"musicgen_{hash(prompt) % 100000}.wav")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": MODEL_NAME})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9881, debug=False)
