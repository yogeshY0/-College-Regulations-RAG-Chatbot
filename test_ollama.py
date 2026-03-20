#!/usr/bin/env python3
"""
Quick test script to verify Ollama is working correctly.
Run: python test_ollama.py
"""

import requests
import json
import sys

def test_ollama_connection():
    """Test if Ollama server is running"""
    print("Testing Ollama connection...")
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        print("✅ Ollama server is running!")
        return True
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to Ollama. Is it running?")
        print("   Solution: Open a terminal and run: ollama serve")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_ollama_models():
    """Check what models are installed"""
    print("\nChecking installed models...")
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        data = response.json()
        models = data.get("models", [])

        if not models:
            print("❌ No models installed!")
            print("   Solution: Run: ollama pull llama3.2")
            return False

        print(f"✅ Found {len(models)} model(s):")
        for model in models:
            name = model.get("name", "Unknown")
            size = model.get("size", 0)
            size_gb = size / (1024**3)
            print(f"   - {name} ({size_gb:.1f} GB)")

        # Check for llama
        model_names = [m.get("name", "") for m in models]
        has_llama = any("llama" in name for name in model_names)

        if has_llama:
            print("✅ llama3.2 is installed!")
            return True
        else:
            print("⚠️  llama3.2 not found (but other models are)")
            print("   Solution: Run: ollama pull llama3.2")
            return False

    except Exception as e:
        print(f"❌ Error checking models: {e}")
        return False

def test_ollama_inference():
    """Test if Ollama can generate a response"""
    print("\nTesting Ollama inference...")
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2",
                "prompt": "Answer briefly: What is 2+2?",
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 50
                }
            },
            timeout=60
        )

        if response.status_code != 200:
            print(f"❌ Ollama returned status {response.status_code}")
            print(f"   Response: {response.text[:200]}")
            return False

        data = response.json()
        answer = data.get("response", "").strip()

        if not answer:
            print("❌ Ollama returned empty response!")
            return False

        print(f"✅ Ollama inference working!")
        print(f"   Question: What is 2+2?")
        print(f"   Answer: {answer[:100]}")
        return True

    except requests.exceptions.Timeout:
        print("❌ Ollama timed out (>60 seconds)")
        print("   The model may be too slow. Try restarting Ollama.")
        return False
    except Exception as e:
        print(f"❌ Inference error: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("OLLAMA DIAGNOSTIC TEST")
    print("=" * 60)

    tests = [
        ("Connection", test_ollama_connection),
        ("Models", test_ollama_models),
        ("Inference", test_ollama_inference),
    ]

    results = []
    for name, test_fn in tests:
        result = test_fn()
        results.append((name, result))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, result in results:
        status = "✅" if result else "❌"
        print(f"{status} {name}")

    all_passed = all(result for _, result in results)

    if all_passed:
        print("\n🎉 All tests passed! Your Ollama setup is working.")
        print("    Run: streamlit run app.py")
        return 0
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")
        print("    Read OLLAMA_SETUP.md for detailed instructions.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

