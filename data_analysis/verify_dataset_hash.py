import hashlib
import os
import json
import warnings
import glob

try:
    import git

    def get_repo_sha1_and_diff(search_parent_directories=False):
        try:
            repo = git.Repo(search_parent_directories=search_parent_directories)
            current_commit_hash = repo.head.commit.hexsha
            diff = repo.head.commit.diff(None, create_patch=True)

            return current_commit_hash, diff

        except Exception as e:
            warnings.warn(f"Git module found but generating git information failed: {type(e)}, {e}")
            return "GIT_FAILED", "GIT_FAILED"

except ImportError:
    warnings.warn("Git module not found. Git information will not be included in metadata."
                  "To solve, run `pip install GitPython` to install it.")

    def get_repo_sha1_and_diff():
        return "NO_GIT_MODULE", "NO_GIT_MODULE"


# Update when changing this script!!!
DATASET_HASH_VERSION = "v0.2.0"


def sha256sum(filename) -> str:
    """
    Hashes a file using SHA256, returns hex string.
    Consistent with command line programs `sha256sum` on Linux and `Get-FileHash` on Windows (Powershell)
    """
    h = hashlib.sha256()
    b = bytearray(128 * 1024)
    mv = memoryview(b)
    with open(filename, 'rb', buffering=0) as f:
        for n in iter(lambda: f.readinto(mv), 0):
            h.update(mv[:n])
    return h.hexdigest()


def hash_all(file_paths):
    result_dict = {}

    for f in file_paths:
        assert os.path.isfile(f), f"{os.path.abspath(f)} is not a file!"
        key = os.path.join(os.path.basename(os.path.dirname(f)), os.path.basename(f))
        key = key.replace(r"\\", "/")  # Windows-Linux compatibility maybe...
        key = key.replace("\\", "/")
        result_dict[key] = sha256sum(f)

    return result_dict


def generate_emsec_metadata(
        glob_pattern,
        probe="UNSPECIFIED", amplifier="UNSPECIFIED",
        dataset_name="UNSPECIFIED", comments="UNSPECIFIED", recursive=True,
        force_wildcard_in_glob_pattern=True,
        **varargs):
    if force_wildcard_in_glob_pattern and glob_pattern[-1] != "*":
        # double star is needed for glob to recurse directories to arbitrary depth
        glob_pattern = os.path.join(glob_pattern, "**")
    all_file_paths = glob.glob(glob_pattern, recursive=recursive)
    file_hash_dict = hash_all([p for p in all_file_paths if os.path.isfile(p)])

    if dataset_name == "UNSPECIFIED":
        dataset_name = f"__GLOB_PATTERN__={glob_pattern}"

    git_hash, git_diff = get_repo_sha1_and_diff(search_parent_directories=True)
    text_git_diff = [str(d) for d in git_diff]

    dataset_info = {
        "GIT_SHA1": git_hash,
        "GIT_DIFF": text_git_diff,
        "DATASET_HASH_VERSION": DATASET_HASH_VERSION,
        "description": "Measurement of electromagnetic emissions from Quantum key distribution electronics. "
                       "Contains measurement setup description and sha256 file hashes of measurement results.",
        "comments": comments,
        "dataset_name": dataset_name,
        "probe": probe,
        "amplifier": amplifier,
        "FILE_SHA256": file_hash_dict,
    }
    dataset_info.update(varargs)
    return dataset_info


def verify_dataset(metadata, glob_pattern, verbose=True):
    true_metadata = generate_emsec_metadata(glob_pattern)

    if true_metadata["DATASET_HASH_VERSION"] != metadata["DATASET_HASH_VERSION"]:
        warnings.warn(f"Validating metadata version {true_metadata['DATASET_HASH_VERSION']} "
                      f"using code {true_metadata['DATASET_HASH_VERSION']}")

    true_file_hashes = true_metadata["FILE_SHA256"]

    for path, filehash in metadata["FILE_SHA256"].items():
        path = path.replace(r"\\", "/")  # Windows-Linux compatibility maybe...
        path = path.replace("\\", "/")
        assert path == "measurement_metadata.json" or \
               true_file_hashes[path] == filehash, \
               f"File {path} has hash {filehash} but expected {true_file_hashes[path]}"
        if verbose:
            print(f"(Verified sha256) {filehash}  {path}")

    return true_metadata


def verify_json_metadata(data_dir, verbose=True):
    with open(os.path.join(data_dir, "measurement_metadata.json"), "r") as f:
        loaded_metadata = json.load(f)
    computed_metadata = verify_dataset(loaded_metadata, data_dir, verbose=verbose)
    return loaded_metadata


def put_metadata_json_into_directory(data_dir, overwrite_existing=False, **varargs):
    """
    data_dir is expected to be directory of directories, each containing data files.
    Perform this after measurement to generate metadata.
    """
    glob_pattern = os.path.join(data_dir, "**")  # two stars -> "recursive" mode matches deep into directories
    metadata_dict = generate_emsec_metadata(
        glob_pattern,
        recursive=True,
        **varargs
    )

    json_path = os.path.join(data_dir, "measurement_metadata.json")
    assert not os.path.exists(json_path) or overwrite_existing, \
        f"Aborting metadata creation: Metadata file {json_path} exists. Set `overwrite_existing` to overwrite."

    with open(json_path, "w+") as f:
        json.dump(metadata_dict, f)

    print(f"Created metadata at {json_path}.")


def main_test():
    data_directory = os.path.join("Data", "1")
    glob_pattern = os.path.join(data_directory, "**")
    dataset_info = generate_emsec_metadata(glob_pattern)
    with open("tmp.json", "w+") as f:
        json.dump(dataset_info, f)

    with open("tmp.json", "r") as f:
        loaded_metadata = json.load(f)

    assert dataset_info == loaded_metadata

    verify_dataset(dataset_info, glob_pattern)

    put_metadata_json_into_directory(data_directory, probe="LK")

    verify_json_metadata(data_directory)


if __name__ == "__main__":
    main_test()
