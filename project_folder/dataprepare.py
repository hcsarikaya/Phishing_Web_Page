import os
import shutil

def prepare_dataset(dataset_path, output_folder):
    phishing_folder = os.path.join(output_folder, "Phishing")
    legitimate_folder = os.path.join(output_folder, "Legitimate")

    # Create output folders if they don't exist
    os.makedirs(phishing_folder, exist_ok=True)
    os.makedirs(legitimate_folder, exist_ok=True)

    # Process Phishing set
    process_category(os.path.join(dataset_path, "phish_sample_30k"), phishing_folder)

    # Process Legitimate set (including Misleading)
    process_category(os.path.join(dataset_path, "benign_25k"), legitimate_folder)
    process_category(os.path.join(dataset_path, "misleading"), legitimate_folder)


def process_category(category_folder, output_folder):
    for filename in os.listdir(category_folder):
        filepath = os.path.join(category_folder, filename)
        for fn in os.listdir(filepath):
            if fn.endswith("html.txt"):
                new_filename = f"{filename}.txt"
                file_path = os.path.join(filepath, fn)
                dest = os.path.join(output_folder, new_filename)
                try:
                    shutil.copy(file_path, dest)
                    print("File copied successfully.")

                # If source and destination are same
                except shutil.SameFileError:
                    print("Source and destination represents the same file.")

                # If there is any permission issue
                except PermissionError:
                    print("Permission denied.")

                # For other errors
                except:
                    print("Error occurred while copying file.")


def main():
    dataset_path = "PhishIntention"
    output_folder = "pythonProject1"

    prepare_dataset(dataset_path, output_folder)



if __name__ == "__main__":
    main()
