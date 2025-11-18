import asyncio
from interrupt_handler import InterruptHandler

# Fake agent with a stop callback
class FakeAgent:
    async def stop(self):
        print(">>> STOP called!")

async def test_interrupt_handler():
    agent = FakeAgent()

    # Initialize handler
    handler = InterruptHandler(
        agent,
        ignored_words=["uh", "umm", "hmm"],
        stop_keywords=["stop", "wait"],
        confidence_threshold=0.5,
        on_stop_callback=agent.stop
    )

    # Simulate agent speaking
    await handler.on_tts_start()

    # Test filler-only transcript (should be ignored)
    result = await handler.on_transcript_event("uh umm")
    print("Filler-only result:", result)

    # Test stop keyword (should trigger stop)
    result = await handler.on_transcript_event("please stop now")
    print("Stop keyword result:", result)

    # Test low-confidence transcript (should ignore)
    result = await handler.on_transcript_event("something random", confidence=0.4)
    print("Low-confidence result:", result)

    # Test meaningful content while agent speaking (should trigger stop)
    result = await handler.on_transcript_event("turn on the lights", confidence=0.9)
    print("Meaningful content result:", result)

    # Simulate agent stops speaking
    await handler.on_tts_end()

    # Test transcript when agent is quiet (should pass through)
    result = await handler.on_transcript_event("hello there", confidence=0.9)
    print("Agent quiet result:", result)


asyncio.run(test_interrupt_handler())
