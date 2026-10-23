import json
import time

import requests

# Paste your NEW Gemini API key here.
API_KEY = "API_KEY_HERE"

MODEL = "gemini-3.5-flash"

API_URL = (
    "https://generativelanguage.googleapis.com/"
    f"v1beta/models/{MODEL}:streamGenerateContent?alt=sse"
)


def ask_gemini(session, messages, debug=False):
    """Send a request and print the answer as it arrives."""

    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": API_KEY,
    }

    body = {
        "systemInstruction": {
            "parts": [
                {
                    "text": (
                        "You are a friendly Python tutor. "
                        "Answer briefly, usually in 2-4 sentences. "
                        "Use short code examples when useful."
                    )
                }
            ]
        },
        "contents": messages,
        "generationConfig": {
            "maxOutputTokens": 512,
            "thinkingConfig": {
                "thinkingLevel": "minimal"
            },
        },
    }

    if debug:
        print("\n--- REQUEST ---")
        print("POST", API_URL)
        print("API key: [HIDDEN]")
        print(json.dumps(body, indent=2))

    answer_parts = []
    model_parts = []
    started = time.perf_counter()
    first_text_received = False
    finish_reason = None

    with session.post(
        API_URL,
        headers=headers,
        json=body,
        stream=True,
        timeout=(10, 60),
    ) as response:

        if debug:
            print("HTTP status:", response.status_code)

        if not response.ok:
            try:
                error = response.json()
                message = error.get("error", {}).get(
                    "message", response.reason
                )
            except ValueError:
                message = response.reason

            raise RuntimeError(
                f"HTTP {response.status_code}: {message}"
            )

        response.encoding = "utf-8"
        print("\nAI: ", end="", flush=True)

        # Read the server-sent events as they arrive.
        for line in response.iter_lines(
            chunk_size=1,
            decode_unicode=True,
        ):
            if not line or not line.startswith("data:"):
                continue

            payload = line[5:].strip()

            if not payload or payload == "[DONE]":
                continue

            event = json.loads(payload)

            if "error" in event:
                raise RuntimeError(
                    event["error"].get(
                        "message", "Streaming API error"
                    )
                )

            candidates = event.get("candidates", [])

            if not candidates:
                continue

            candidate = candidates[0]
            finish_reason = candidate.get(
                "finishReason", finish_reason
            )

            parts = candidate.get("content", {}).get(
                "parts", []
            )

            for part in parts:
                # Preserve full parts for conversation history.
                model_parts.append(part)

                text = part.get("text", "")

                if text and not part.get("thought", False):
                    if not first_text_received:
                        first_text_received = True

                        if debug:
                            elapsed = time.perf_counter() - started
                            print(
                                f"[First text: {elapsed:.2f}s] ",
                                end="",
                                flush=True,
                            )

                    print(text, end="", flush=True)
                    answer_parts.append(text)

        print()

    if not answer_parts:
        raise RuntimeError(
            "No text received. "
            f"Finish reason: {finish_reason or 'unknown'}"
        )

    if finish_reason == "MAX_TOKENS":
        print("[Response reached the length limit.]")

    if debug:
        elapsed = time.perf_counter() - started
        print(f"Total time: {elapsed:.2f}s")

    return {
        "role": "model",
        "parts": model_parts,
    }


def main():
    if API_KEY == "PASTE_YOUR_NEW_GEMINI_KEY_HERE":
        print("Put your new Gemini API key in API_KEY first.")
        return

    history = []
    debug = False

    print("\n=== StudyBuddy: Python API Chat ===")
    print("/clear - reset conversation")
    print("/debug - show request details and timing")
    print("/quit  - exit")

    # Reuse the HTTP connection between messages.
    with requests.Session() as session:
        while True:
            try:
                message = input("\nYou: ").strip()

                if message.lower() in ("/quit", "quit", "exit"):
                    print("Goodbye!")
                    break

                if message.lower() == "/clear":
                    history.clear()
                    print("Conversation cleared.")
                    continue

                if message.lower() == "/debug":
                    debug = not debug
                    print("Debug:", "ON" if debug else "OFF")
                    continue

                if not message:
                    continue

                user_message = {
                    "role": "user",
                    "parts": [{"text": message}],
                }

                pending_history = history + [user_message]

                model_message = ask_gemini(
                    session,
                    pending_history,
                    debug,
                )

                # Only save successful exchanges.
                history = pending_history + [model_message]

            except requests.exceptions.Timeout:
                print("\nRequest timed out. Try again.")

            except requests.exceptions.RequestException:
                print("\nConnection error. Check your internet.")

            except (RuntimeError, ValueError) as error:
                print("\nError:", error)

            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye!")
                break


if __name__ == "__main__":
    main()