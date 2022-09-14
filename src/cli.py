from pathlib import Path
from fire import Fire


_data_dir = Path(__file__).parent / "data"


class CLI:
    @staticmethod
    def clean():
        """
        TODO: explain what I'm doing
        """
        from clean import clean_csv

        clean_csv(
            file_path=_data_dir / "tweets.csv",
            out_path=_data_dir / "tweets_cleaned.csv",
        )

    @staticmethod
    def sent():
        """
        TODO: explain what I'm doing
        """
        from sent import sentiment_csv

        sentiment_csv(
            file_path=_data_dir / "tweets_cleaned.csv",
            out_path=_data_dir / "tweets_analyzed.csv",
        )

    @staticmethod
    def bow(file_path: Path):
        """
        TODO: explain what I'm doing
        """
        from bow import bow_json

        bow_json(file_path=_data_dir / "tweets_analyzed.csv")


if __name__ == "__main__":
    Fire(CLI)
