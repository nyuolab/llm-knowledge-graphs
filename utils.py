# NYU OLAB, 2023

# imports
import json
import argparse
import string
import hashlib
import random

import torch
import omegaconf
import spacy
from negspacy.negation import Negex  # for pipe, not actually referenced


# NLP to check if a relationship is positive
def load_negative_pipeline():
    nlp = spacy.load("en_core_sci_md")
    nlp.add_pipe("negex")

    return nlp


def is_positive(triplet, pipeline):
    doc = pipeline(triplet[1]) if type(triplet) == list else pipeline(triplet)
    negations = [i._.negex for i in doc.ents]
    return sum(negations) % 2 == 0  # if even -> no negation, if odd -> negation


# pretty print
def pretty_print_omegaconf(cfg):
    """
    Pretty print an omegaconf with multiple levels.
    """
    print(omegaconf.OmegaConf.to_yaml(cfg))


# read json
def json_io(path: str, read: bool = True) -> dict | list[dict]:
    """
    Handle json input/output. Also works with jsonl.
    """
    if read:
        if path.endswith(".json"):
            with open(path, "r") as f:
                return json.load(f)
        elif path.endswith(".jsonl"):
            with open(path, "r") as f:
                return [json.loads(line) for line in f]
        else:
            raise ValueError("json_io: path must end in .json or .jsonl")
    else:
        if path.endswith(".json"):
            with open(path, "w") as f:
                json.dump(f)
        elif path.endswith(".jsonl"):
            with open(path, "w") as f:
                for line in f:
                    json.dump(line, f)
        else:
            raise ValueError("json_io: path must end in .json or .jsonl")


# set seeds
def set_seeds(seed: int = 0, cudnn_deterministic: bool = True):
    """
    Set seeds for reproducibility.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = cudnn_deterministic


# get omegaconf config
def get_config(path: str) -> omegaconf.dictconfig.DictConfig:
    """
    Get omegaconf config from yaml file.
    """
    return omegaconf.OmegaConf.load(path)


# hash a string
def hash_string(s: str):
    return hashlib.sha256(s.encode()).hexdigest()


# hash args
def hash_args(*args, **kwargs):
    hasher = hashlib.sha256()

    # Update the hash with the arguments
    for arg in args:
        hasher.update(repr(arg).encode("utf-8"))

    # Update the hash with the keyword arguments
    for key, value in kwargs.items():
        hasher.update(f"{key}={repr(value)}".encode("utf-8"))

    # Return the hexadecimal digest of the hash
    return hasher.hexdigest()


# string encode/decode in ascii
def string_encode(s: str):
    return s.encode("ascii", "ignore").decode()


# strip whitespace and punctuation from string
def strip_whitespace_and_punctuation(s: str):
    # warning: this will also string newlines
    return s.strip(string.punctuation + string.whitespace)


# string preprocess
def string_preprocess(s: str):
    if type(s) != str:
        s = str(s)  # convert to string and warn
        print(f"Warning: string_preprocess: s is not a string: {s}")
    # s = s.lower()
    s = s.replace("_", " ")
    s = s.strip(string.whitespace + string.punctuation)
    return s


# parse arguments
def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Active defense against medical harm using ontology-enhanced review of medical named entities."
    )

    # config
    parser.add_argument(
        "--config",
        type=str,
        default="./config.yaml",
        help="Path to yaml config file (default: config.yaml)",
    )

    # parse
    args = parser.parse_args()

    return args


# check if relationship is in list
def relationship_in_list(
    relationship: str, relationship_list: list[str], relationship_metadatas: list[dict]
) -> (list[str], list[dict]):
    # preprocess
    relationship_processed = string_preprocess(relationship).lower()
    # check if relationship is in relationship_list
    for idx, r in enumerate(relationship_list):
        r_processed = string_preprocess(r).lower()
        if relationship_processed == r_processed:
            return [r], [relationship_metadatas[idx]]

    return relationship_list, relationship_metadatas


# check if a concept is in list
def concept_in_list(concept: str, concept_list: list[str]) -> list[str]:
    # preprocess
    concept_processed = string_preprocess(concept).lower()

    # check if concept is in concept_list
    for c in concept_list:
        c_processed = string_preprocess(c).lower()
        if concept_processed == c_processed:
            return [c]

    return concept_list


# medical phrase class
class MedicalPhrase:
    """
    Stores a medical phrase or 'triplet' (origin, relationship, target).
    """

    def __init__(
        self,
        origin: str,
        relationship: str,
        target: str,
        origin_uid: str = None,
        target_uid: str = None,
    ):
        self.origin = origin
        self.relationship = relationship
        self.target = target

        # hotfix
        if "ddx" in self.relationship.lower():
            # replace by index but keep case
            self.relationship = self.relationship.lower().replace(
                "ddx", "differential diagnosis"
            )

        # uid
        self.origin_uid = origin_uid
        self.target_uid = target_uid

    def __repr__(self):
        return f"{self.origin}\n{self.relationship}\n{self.target}"

    def __str__(self):
        return f"{self.origin} {self.relationship} {self.target}"
