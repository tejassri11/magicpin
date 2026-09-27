"""
prompts/reply_prompts.py — Prompt variants for multi-turn reply composition.

These are used by reply_handler.py when the bot needs to respond to a merchant
reply (action mode, curveball handling, graceful exit, etc.).
"""
from __future__ import annotations

REPLY_NORMAL = """
CONTEXT: You are continuing an existing conversation with this merchant.
The conversation history is shown below. The merchant just replied.

TASK:
- Respond naturally to what the merchant said
- Advance the conversation toward the trigger's goal
- Don't repeat what you already said
- Keep it short (WhatsApp — not email)
- If the merchant asked a specific question, answer it first before pivoting
- End with a single, clear next step (the CTA)

Output ONLY JSON: {"action": "send", "body": "...", "cta": "...", "rationale": "..."}
"""

REPLY_ACTION_MODE = """
CONTEXT: The merchant has explicitly committed ("ok let's do it", "yes go ahead", etc.)
This is an INTENT TRANSITION — switch from qualifying to ACTION immediately.

TASK:
- DO NOT ask another qualifying question
- Start executing the thing you offered: draft the content, schedule the post, confirm the action
- Be concrete: what exactly will happen, what does the merchant need to confirm
- State the scope: how many patients, what will be sent, by when
- CTA: binary confirm/cancel

Output ONLY JSON: {"action": "send", "body": "...", "cta": "binary_yes_no", "rationale": "..."}
"""

REPLY_CURVEBALL = """
CONTEXT: The merchant asked about something outside Vera's scope (GST, legal, salary, etc.)

TASK:
- Politely acknowledge you can't help with that specific topic (1 sentence)
- Immediately redirect back to the original conversation topic
- Don't apologise excessively — just redirect cleanly
- CTA: pick up where the original conversation left off

Output ONLY JSON: {"action": "send", "body": "...", "cta": "...", "rationale": "..."}
"""

REPLY_AUTO_PROBE = """
CONTEXT: The bot detected what appears to be a WhatsApp Business auto-reply.
The auto-reply pattern suggests the merchant's phone is set to auto-respond.

TASK:
- Acknowledge gently that this looks like an automated reply
- Leave a clear instruction for the OWNER (the human) when they see the message
- Keep it very short (1-2 lines)
- CTA: binary_yes_no ("When you see this, just reply YES to proceed")

Output ONLY JSON: {"action": "send", "body": "...", "cta": "binary_yes_no", "rationale": "..."}
"""

REPLY_GRACEFUL_EXIT = """
CONTEXT: This is the bot's final message before ending the conversation.
The merchant may have been unresponsive, expressed mild disinterest, or the turn limit was reached.

TASK:
- Close gracefully — no pressure, no guilt
- Leave a re-entry path ("If anything changes, just reply 'Hi Vera'")
- One sentence max
- CTA: none

Output ONLY JSON: {"action": "send", "body": "...", "cta": "none", "rationale": "..."}
"""
