from pathlib import Path
from fire import Fire
from tabulate import tabulate


_data_dir = Path(__file__).parent / "data"


class CLI:
    """
    Command Line Interface for Robert's Bachelor Thesis
    """

    @staticmethod
    def clean():
        """
        TODO: explain what I'm doing

        - alle satzzeichen wegschmeissen
        - bla
        - blub
        """
        from clean import clean_csv

        clean_csv(
            file_path=_data_dir / "tweets.csv",
            out_path=_data_dir / "tweets_cleaned.csv",
        )

    @staticmethod
    def sent():
        """
        file1 -> sent methoden bla bla -> file2
        """
        from sent import sentiment_csv

        sentiment_csv(
            file_path=_data_dir / "tweets_cleaned.csv",
            out_path=_data_dir / "tweets_analyzed.csv",
        )

    @staticmethod
    def bow():
        """
        get accuracy score for tweets_analyzed.csv
        """
        from bow import bow_csv

        summary = bow_csv(file_path=_data_dir / "tweets_analyzed.csv")
        print(tabulate(list(summary.items()), headers=['Stat', 'Value']))


if __name__ == "__main__":
    Fire(CLI)
