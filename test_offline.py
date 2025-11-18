# test_offline.py
import asyncio
from types import SimpleNamespace
from interrupt_handler import InterruptHandler

class OfflineInterruptTester:
    def __init__(self):
        # InterruptHandler with mock agent
        self.interrupt_handler = InterruptHandler(
            agent=SimpleNamespace(),
            ignored_words=["uh", "umm", "hmm", "haan"],
            stop_keywords=["stop", "wait", "hold on", "no"],
            confidence_threshold=0.6,
            on_stop_callback=self.stop
        )

    async def stop(self):
        print(">>> STOP called!")

    async def run(self, transcripts):
        for t in transcripts:
            # Simulate agent speaking if specified
            if t.get("agent_speaking", False):
                await self.interrupt_handler.on_tts_start()
            else:
                await self.interrupt_handler.on_tts_end()

            # Process transcript
            result = await self.interrupt_handler.on_transcript_event(
                transcript=t["transcript"],
                confidence=t["confidence"]
            )
            print(result)

# ----- Offline Demo -----
async def main():
    gen = OfflineInterruptTester()

    transcripts = [
        {"transcript": "uh umm", "confidence": 0.9, "agent_speaking": True},                # filler while speaking
        {"transcript": "please stop now", "confidence": 0.95, "agent_speaking": True},     # stop keyword
        {"transcript": "something random", "confidence": 0.7, "agent_speaking": False},    # agent quiet
        {"transcript": "turn on the lights", "confidence": 0.8, "agent_speaking": False},  # normal content
        {"transcript": "wait I forgot", "confidence": 0.85, "agent_speaking": True},      # stop keyword in mixed content
        {"transcript": "umm I think yes", "confidence": 0.55, "agent_speaking": True},    # low confidence filler + speaking
        {"transcript": "hello", "confidence": 0.3, "agent_speaking": True},               # very low confidence
        {"transcript": "no", "confidence": 0.9, "agent_speaking": True},                  # single stop word
    ]

    await gen.run(transcripts)

if __name__ == "__main__":
    asyncio.run(main())
