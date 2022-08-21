from pathlib import Path
from fire import Fire

from clean import clean_json
from sentiment import sent_json


class CLI:
    def clean(self, file_path: Path):
        """
        Clean the tweets
        (Warning: Make a backup of the file if you want to keep it)
        """
        clean_json(file_path=file_path)

    def sent(self, file_path: Path, method: str = "all"):
        """
        Analyze the sentiment of the tweets.
        (Default run ALL methods sequentually)"""
        sent_json(file_path=file_path, method_id=method)

    def all(self, file_path: Path):
        """
        Clean and analyze the sentiment of the tweets.
        """
        self.clean(file_path=file_path)
        self.sent(file_path=file_path, method="all")


if __name__ == "__main__":
    Fire(CLI)
