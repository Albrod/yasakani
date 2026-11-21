"""0.3.1: the options of the Qwen3.8 engine that came with #26 (--suppress-tokens) and #27 (--yarn2) are refused for Inkling-2 model
folders instead of being ignored without a word."""
import argparse
import contextlib
import io
import os
import unittest
from unittest import mock

from yasakani import cli, inkling2_cli
from yasakani._native import YasakaniError


class QwenOnlyOptionsTests(unittest.TestCase):
    def test_suppress_tokens_refused(self):
        for cmd in ("serve", "chat"):
            args = cli.build_parser().parse_args([cmd, "--suppress-tokens", "mask.json"])
            with self.assertRaisesRegex(YasakaniError, "--suppress-tokens"):
                inkling2_cli.refuse_unsupported(args)

    def test_yarn2_refused(self):
        for cmd in ("serve", "chat", "bench"):
            args = cli.build_parser().parse_args([cmd, "--yarn2"])
            with self.assertRaisesRegex(YasakaniError, "--yarn2"):
                inkling2_cli.refuse_unsupported(args)

    def test_plain_options_pass(self):
        for cmd in ("serve", "chat", "bench", "selfcheck"):
            inkling2_cli.refuse_unsupported(cli.build_parser().parse_args([cmd]))

    def test_suppress_env_noted_once(self):
        args = argparse.Namespace()
        err = io.StringIO()
        with mock.patch.dict(os.environ, {"YASAKANI_SUPPRESS_TOKENS": "mask.json"}), contextlib.redirect_stderr(err):
            inkling2_cli.refuse_unsupported(args)
            inkling2_cli.refuse_unsupported(args)
        self.assertEqual(err.getvalue().count("ignoring YASAKANI_SUPPRESS_TOKENS"), 1)


if __name__ == "__main__":
    unittest.main()
