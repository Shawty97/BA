import csv
import json
from tqdm import tqdm
from pathlib import Path

NEGATIVE_UPPER_BOUND = -0.3
NEUTRAL_UPPER_BOUND = 0.3


def merge_company_status_files(
    file_companies: Path, file_analyzed: Path, out_path: Path
):
    new_company_info_vectors: dict[list] = dict()

    # load company data
    with open(file_companies, encoding="utf-8") as file_in:
        reader = csv.reader(file_in, delimiter=",")
        _ = next(reader)

        print("Step 1: Load company data")
        for row in tqdm(reader):
            new_company_info_vectors[row[2]] = [[], [], [], [], -1, -1, -1]

    # load tweets for the companies
    with open(file_analyzed, encoding="utf-8") as file_in:
        reader = csv.reader(file_in, delimiter=",")
        _ = next(reader)

        print("Step 2: Load tweets")
        for row in tqdm(reader):
            company_name = row[18]
            if company_name in new_company_info_vectors:
                # sent vader
                new_company_info_vectors[company_name][0].append(float(row[-1]))

                # retweet count
                new_company_info_vectors[company_name][1].append(int(row[8]))

                # like count
                new_company_info_vectors[company_name][2].append(int(row[10]))

                # reply count
                new_company_info_vectors[company_name][3].append(int(row[11]))
    
    
    # calculate sums
    for company in new_company_info_vectors.values():
        # positive tweets
        company[4] = len([x for x in company[0] if x >= NEUTRAL_UPPER_BOUND])

        # neutral tweets
        company[6] = len([x for x in company[0] if x >= NEGATIVE_UPPER_BOUND and x < NEUTRAL_UPPER_BOUND])

        # negative tweets
        company[5] = len([x for x in company[0] if x < NEGATIVE_UPPER_BOUND])

    # load company data
    with open(file_companies, encoding="utf-8") as file_in:
        reader = csv.reader(file_in, delimiter=",")
        header = next(reader)

        print("Step 3: Write merged data")
        # write merged data
        with open(out_path, mode="w", encoding="utf-8", newline="") as file_out:
            writer = csv.writer(file_out, delimiter=",")
            writer.writerow(
                header
                + [
                    "sent_vader_vector",
                    "retweet_counts",
                    "like_counts",
                    "reply_counts",
                    "pos_tweet_count",
                    "neg_tweet_count",
                    "neutr_tweet_count",
                ]
            )

            for row in tqdm(reader):
                # ignore rows where we couldn't match a company
                if not any(new_company_info_vectors.get(row[2])):
                    continue

                writer.writerow(row + new_company_info_vectors.get(row[2]))
