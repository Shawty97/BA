import csv
import json
from tqdm import tqdm
from pathlib import Path


def merge_company_status_files(
    file_companies: Path, file_analyzed: Path, out_path: Path
):
    new_company_info_vectors: dict[list] = dict()

    # load company data
    with open(file_companies, encoding="utf-8") as file_in:
        reader = csv.reader(file_in, delimiter=",")
        _ = next(reader)

        for row in reader:
            new_company_info_vectors[row[2]] = [[], [], [], []]

    # load tweets for the companies
    with open(file_analyzed, encoding="utf-8") as file_in:
        reader = csv.reader(file_in, delimiter=",")
        _ = next(reader)

        for row in reader:
            company_name = row[18]
            if company_name in new_company_info_vectors:
                # sent vader
                new_company_info_vectors[company_name][0].append(row[-1])

                # retweet count
                new_company_info_vectors[company_name][1].append(row[8])

                # like count
                new_company_info_vectors[company_name][2].append(row[10])

                # reply count
                new_company_info_vectors[company_name][3].append(row[11])

    # for company in new_company_info_vectors.values():
    #     for i in range(len(company)):
    #         company[i] = json.dumps(company[i]).replace(",", " ")

    # load company data
    with open(file_companies, encoding="utf-8") as file_in:
        reader = csv.reader(file_in, delimiter=",")
        header = next(reader)

        # write merged data
        with open(out_path, mode="w", encoding="utf-8", newline="") as file_out:
            writer = csv.writer(file_out, delimiter=",")
            writer.writerow(
                header
                + ["sent_vader_vector", "retweet_counts", "like_counts", "reply_counts"]
            )

            for row in tqdm(reader):
                if not any(new_company_info_vectors.get(row[2])):
                    continue

                writer.writerow(row + new_company_info_vectors.get(row[2]))
