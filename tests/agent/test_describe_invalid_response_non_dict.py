"""``describe_invalid_response`` must diagnose a non-response, never crash on it.

Regression for the silent cron failures (kanban t_dcc8d8c7): a provider callback returned an
un-awaited coroutine where a chat-completion response belongs, and the unguarded ``vars(response)``
in this diagnostic raised ``TypeError: vars() argument must have __dict__ attribute`` — replacing
the real "invalid response" failure with a misleading provider error.
"""
import asyncio
import unittest

from agent.turn_recovery import describe_invalid_response


class _StubAgent:
    api_mode = "chat_completions"
    provider = "claude-subscription-directsdk-experimental"
    verbose_logging = False


class DescribeInvalidResponseNonDictTest(unittest.TestCase):
    def test_unawaited_coroutine_response_is_diagnosed_not_raised(self):
        async def _noop():
            return None

        coro = _noop()
        try:
            error_msg, provider_name, _hint = describe_invalid_response(_StubAgent(), coro, 0.1)
        finally:
            coro.close()
        self.assertEqual(error_msg, "Unknown")
        self.assertEqual(provider_name, "Unknown")

    def test_object_response_attributes_are_still_reported(self):
        class _Resp:
            model = None
            note = "kept"

        error_msg, provider_name, _hint = describe_invalid_response(_StubAgent(), _Resp(), 0.1)
        self.assertEqual(error_msg, "Unknown")
        self.assertEqual(provider_name, "Unknown")

    def test_dict_response_is_diagnosed_without_raising(self):
        error_msg, provider_name, _hint = describe_invalid_response(_StubAgent(), {"error": None}, 0.1)
        self.assertEqual(error_msg, "Unknown")
        self.assertEqual(provider_name, "Unknown")


if __name__ == "__main__":
    unittest.main()
