import unittest

import inventory_checkpoint_prompts as inventory


class InventoryTest(unittest.TestCase):
    def test_extracts_only_unambiguous_marked_prompts_without_assigning_labels(self):
        first, second, third = (character * 40 for character in "abc")
        message = "Assistant checkpoint\n\nUser prompt:\n\nPlease inspect this.\n\nReplit-Commit-Author: Assistant\nReplit-Commit-Session-Id: session-one\n"
        ambiguous = "User prompt:\n\nUnclear trailer.\n\nReplit-Commit-Author: Assistant\nExtra text\n"
        raw = (f"{first}\0date\0{message}\0\n{second}\0date\0{message}\0\n"
               f"{third}\0date\0{ambiguous}\0\n").encode()
        report, rows = inventory.parse_history(raw, "project", first)
        self.assertEqual((report["marked_commits"], report["extracted"], report["format_rejected"]), (3, 2, 1))
        self.assertEqual(report["format_rejected_commits"], [third])
        self.assertEqual(report["duplicate_exact_texts"], 1)
        self.assertEqual(report["distinct_sessions"], 1)
        self.assertEqual(rows[0]["text"], "Please inspect this.")
        self.assertIsNone(rows[0]["labels"])
        self.assertIsNone(rows[0]["split"])
        self.assertFalse(rows[0]["verified"])
        self.assertTrue(rows[1]["exact_text_seen_earlier"])

    def test_rejects_partial_stream(self):
        with self.assertRaises(ValueError):
            inventory.parse_history(b"a\0b\0", "project", "a" * 40)


if __name__ == "__main__":
    unittest.main()
