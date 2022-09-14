from pathlib import Path
from fire import Fire



_data_dir = Path(__file__).parent / 'data'


class CLI:
    @staticmethod
    def clean():
        from clean import clean_csv
        clean_csv(
            file_path=_data_dir / 'tweets.csv',
            out_path=_data_dir / 'tweets_cleaned.csv',
        )

    @staticmethod
    def sent(file_path: Path):
        from sentiment import sentiment_json
        file_path = _data_dir / 'tweets_cleaned.csv'
        sentiment_json(file_path=file_path)

    @staticmethod
    def bow(file_path: Path):
        from bow import bow_json
        bow_json(file_path=file_path)


if __name__ == "__main__":
    Fire(CLI)
