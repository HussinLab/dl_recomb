import os
from google.cloud import storage


def get_dataset_from_bucket(bucket_name, dataset_name, save_to_path):
    """[summary]

    Args:
        bucket_name ([type]): [description]
        dataset_name ([type]): [description]
        save_to_path ([type]): [description]

    Returns:
        [type]: [description]
    """

    bucket = storage.Client().bucket(bucket_name)
    blob = bucket.blob(dataset_name)

    fname = dataset_name.split("/")[-1]
    saved_path = os.path.join(save_to_path, fname)

    blob.download_to_filename(saved_path)

    print(f"File download to {saved_path}")
    return saved_path
