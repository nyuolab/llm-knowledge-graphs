# NYU OLAB, 2023

"""
Inference-time screening defense against harm using ontology-enhanced review of medical named entities.

This module contains the main active defense function, which consumes a medical phrase or 'triplet' (origin, relationship, target)
and returns a boolean indicating whether the triplet is valid or not, according to the provided knowledge graph ground truth.
"""

# imports
from utils import *

import omegaconf
import pandas as pd
from tqdm import tqdm

from vector_db import VectorDatabase, KnowledgeGraph


def verify_medical_phrase(
    vector_db: VectorDatabase,
    knowledge_graph: KnowledgeGraph,
    phrase: MedicalPhrase,
    k_r: int = 1,
    k_c: int = 1,
    check_negatives: bool = False,
    check_exact_relationships: bool = False,
    check_exact_concepts: bool = False,
    negatives_pipeline=None,
) -> bool:
    """
    Score a medical phrase or 'triplet' (origin, relationship, target) according to the Knowledge Graph.

    Returns a boolean indicating whether the triplet is valid or not.

    1. Match phrase.relationship to relationship (e.g. "treatment for" --> "treats")
    2. Match phrase.origin to concept/atom (e.g. "lasix" --> "furosemide") for which the relationship from (1) is an outgoing edge
    3. Match phrase.target to concept/atom (e.g. "edema" --> "edema") for which the relationship from (1) is an incoming edge
    4. Return True if there exists a path (1) --> (2) --> (3) in the Knowledge Graph.

    Not Yet Implemented: Extend to multi-step walks (e.g. "furosemide" --> "is a" --> "diuretic" --> "treats" --> "pulmonary edema")

    :param phrase: MedicalPhrase object to score (origin, relationship, target)
    :param k_r: number of relationships to consider for scoring (default 1)
    :param k_c: number of concepts to consider for scoring (per matched relationship; default 1)
    """

    # 1. Match phrase.relationship to relationship (e.g. "treatment for" --> "treats")
    matched_relationships_query = vector_db.match_relationship(
        phrase.relationship, vector_db.relationship_collection.count()
    )
    matched_relationships_tentative = matched_relationships_query["documents"][0]
    matched_relationship_metadatas_tentative = matched_relationships_query["metadatas"][
        0
    ]

    matched_relationships = []
    matched_relationship_metadatas = []
    exclude = []
    filter_idx = 0

    while (
        filter_idx < len(matched_relationships_tentative)
        and len(matched_relationships) < k_r
    ):
        # get relationship
        relationship = matched_relationships_tentative[filter_idx]
        relationship_metadata = matched_relationship_metadatas_tentative[filter_idx]

        # check if relationship is in exclude
        if relationship not in exclude:
            # if not, add to matched_relationships
            matched_relationships.append(relationship)
            matched_relationship_metadatas.append(relationship_metadata)

        # increment filter_idx
        filter_idx += 1

    if check_exact_relationships:
        matched_relationships, matched_relationship_metadatas = relationship_in_list(
            phrase.relationship, matched_relationships, matched_relationship_metadatas
        )

    if check_negatives:
        positive_relationship = is_positive(phrase.relationship, negatives_pipeline)
    else:
        positive_relationship = True

    # Iterate over matched relationships
    result = False

    # Iterate over matched relationships
    for matched_relationship, matched_relationship_metadata in zip(
        matched_relationships, matched_relationship_metadatas
    ):
        # 2. Match phrase.origin to concept/atom (e.g. "lasix" --> "furosemide") for which the relationship from (1) is an outgoing edge
        matched_origins = vector_db.match_concept(
            phrase.origin,
            k_c,
            matched_relationship_metadata["origins"],
        )["documents"][0]

        if check_exact_concepts:
            matched_origins = concept_in_list(phrase.origin, matched_origins)

        # 3. Match phrase.target to concept/atom (e.g. "edema" --> "edema") for which the relationship from (1) is an incoming edge
        matched_targets = vector_db.match_concept(
            phrase.target,
            k_c,
            matched_relationship_metadata["targets"],
        )["documents"][0]

        if check_exact_concepts:
            matched_targets = concept_in_list(phrase.target, matched_targets)

        # 4. Return True if there exists a path (1) --> (2) --> (3) in the Knowledge Graph.
        result, valid_phrase = knowledge_graph.query(
            matched_origins, matched_targets, matched_relationship
        )

        # If result is True, break out of loop
        if result:
            break
        else:
            valid_phrase = ""

    # If positive relationship, return result
    if not positive_relationship:
        result = not result

    # Return result
    return result, valid_phrase


def defense(config, vector_db, knowledge_graph, medical_phrases):
    # Load nlp pipeline if necessary
    if config.verify.check_negatives:
        negative_pipeline = load_negative_pipeline()
    else:
        negative_pipeline = None

    # Iterate over medical phrases
    for row_idx in tqdm(range(len(medical_phrases))):
        # get row
        row = medical_phrases.iloc[row_idx]

        # Create MedicalPhrase object
        phrase = MedicalPhrase(
            row["origin"],
            row["relationship"],
            row["target"],
        )

        # Query knowledge graph for phrase
        result, valid_phrases = verify_medical_phrase(
            vector_db,
            knowledge_graph,
            phrase,
            config.verify.k_r,
            config.verify.k_c,
            config.verify.check_negatives,
            config.verify.check_exact_relationships,
            config.verify.check_exact_concepts,
            negative_pipeline,
        )

        medical_phrases.loc[row_idx, "result"] = result
        medical_phrases.loc[row_idx, "valid_phrase"] = valid_phrases

        # convert result to string
        result_to_string = {
            True: "Non-harmful",
            False: "Harmful",
        }

        # print the phrase and result
        print(
            f"Phrase: {phrase.origin} -- {phrase.relationship} -- {phrase.target} -- Result: {result_to_string[result]}"
        )

    return medical_phrases


# setup knowledge graph and vector database
def setup_kg_vector_db(config):
    # Load ground truth json
    ground_truth = json_io(config.knowledge_graph.ground_truth)

    # Initialize knowledge graph
    knowledge_graph = KnowledgeGraph(ground_truth)

    # Initialize vector database
    vector_db = VectorDatabase(
        config.knowledge_graph.vector_db_root,
        knowledge_graph.relationships,
        knowledge_graph.concepts,
        config.knowledge_graph.embedding_model,
        config.knowledge_graph.embedding_batch_size,
    )

    return knowledge_graph, vector_db


if __name__ == "__main__":
    # Args
    args = parse_args()

    # Load config
    config = omegaconf.OmegaConf.load(args.config)

    # Pretty print config
    pretty_print_omegaconf(config)

    # Set seed
    set_seeds(config.seed)

    # load knowledge graph and vector database
    global knowledge_graph, vector_db
    knowledge_graph, vector_db = setup_kg_vector_db(config)

    # Load medical phrases dataframe
    medical_phrases = pd.read_csv(config.medical_phrases)

    # Check if "result" and "valid_phrase" columns exist, add blank columns if not
    if "result" not in medical_phrases.columns:
        medical_phrases["result"] = False

    if "valid_phrase" not in medical_phrases.columns:
        medical_phrases["valid_phrase"] = ""

    # Main
    medical_phrases = defense(config, vector_db, knowledge_graph, medical_phrases)

    # Save medical phrases dataframe
    medical_phrases.to_csv(config.medical_phrases[:-4] + "_scored.csv", index=False)
