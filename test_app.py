import unittest

from app import is_valid_youtube_url


class YoutubeUrlValidationTests(unittest.TestCase):
    def test_valid_youtube_url(self):
        self.assertTrue(is_valid_youtube_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ"))
        self.assertTrue(is_valid_youtube_url("https://youtu.be/dQw4w9WgXcQ"))

    def test_invalid_url(self):
        self.assertFalse(is_valid_youtube_url("https://example.com/video.mp4"))
        self.assertFalse(is_valid_youtube_url("not a url"))


if __name__ == "__main__":
    unittest.main()
