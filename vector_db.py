# NYU OLAB, 2023

# imports
import os
import shutil

import chromadb
from tqdm import tqdm

from embeddings import *
from utils import *


# Vector Database class
class VectorDatabase:
    """
    Class for Vector Database construction and retrieval.
    Stores two databases: one for concepts, another for relationships.
    """

    def __init__(
        self,
        db_root: str,
        relationships: dict,
        concepts: dict,
        embedding_model: str = "medcpt",
        embedding_batch_size: int = 512,
    ):
        # delete existing database if it exists
        if os.path.exists(db_root):
            shutil.rmtree(db_root)
            os.makedirs(db_root)

        # initialize persistent database
        self.client = chromadb.PersistentClient(db_root)

        # initialize concept/atom collection
        self.embedding_function = self._get_embedding_function(embedding_model)
        self.sentiment_embedding_function = ClinicalBertEmbedding()
        self.batch_size = embedding_batch_size
        self.relationship_collection = self._initialize_relationship_collection(
            relationships
        )
        self.concept_collection = self._initialize_concept_collection(concepts)

        # print all rels
        self.relationships = set(self.relationship_collection.get()["documents"])
        print("Relationships: ", self.relationships)

    def _get_embedding_function(self, embedding_model: str = "medcpt"):
        """
        Get embedding function from embedding model name.
        """
        if embedding_model in EMBED_MODEL_DICT:
            return EMBED_MODEL_DICT[embedding_model]
        else:
            raise NotImplementedError(
                f"embedding model {embedding_model} not implemented"
            )

    def _initialize_get_collection(
        self,
        collection_name: str,
        ef: chromadb.EmbeddingFunction = None,
        delete_existing: bool = False,
    ):
        """
        Initialize collection.
        """
        if ef is None:
            ef = self.embedding_function

        if delete_existing:
            # delete existing collection if it exists
            try:
                collection = self.client.delete_collection(collection_name)
            except:
                pass

        collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )

        return collection

    def target_match_collection(
        self,
        valid_targets: list[str],
        target_to_match: str,
    ):
        """
        Match a target to a list of valid targets.
        """
        # get or create concept/atom collection
        target_match_collection = self._initialize_get_collection(
            "target_match", self.embedding_function, delete_existing=True
        )

        # initialize list of entries
        target_match_collection.add(
            documents=valid_targets,
            ids=[hash_string(t) for t in valid_targets],
        )

        # query
        results = target_match_collection.query(
            query_texts=[string_preprocess(target_to_match)],
            n_results=1,
            include=["documents", "distances"],
        )

        matched = results["documents"][0][0]
        distance = results["distances"][0][0]

        return matched, distance

    def _initialize_concept_collection(self, concepts: dict):
        """
        Initialize concept collection.
        """
        # get or create concept/atom collection
        concept_collection = self._initialize_get_collection("concepts")

        # check size
        if concept_collection.count() >= 1:
            print(
                f"Initializing concept collection already containing {concept_collection.count()} entries..."
            )
            return concept_collection

        # add concepts/atoms to collection
        entries = []
        uids = []
        print("Adding concepts/atoms to collection...")
        for k, v in tqdm(concepts.items()):
            # add entry
            entries.append(string_preprocess(k))

            # add uids
            uids.append({"uids": " ".join(v)})

        # make hashes for id
        hashes = [hash_string(entry) for entry in entries]

        # add entries to collection with uids as metadata, hashes as id for chromadb
        # iterate by self.batch_size to avoid memory issues
        for i in tqdm(range(0, len(entries), self.batch_size)):
            # get batch
            batch = entries[i : i + self.batch_size]
            batch_uids = uids[i : i + self.batch_size]
            batch_hashes = hashes[i : i + self.batch_size]

            try:
                # add batch to collection
                concept_collection.add(
                    documents=batch,
                    ids=batch_hashes,
                    metadatas=batch_uids,
                )
            except Exception as e:
                print(f"Failed to add batch to collection: {e}")
                raise e

        return concept_collection

    def _initialize_relationship_collection(self, relationships: dict):
        """
        Initialize relationship collection.
        """
        # get or create relationship collection
        relationship_collection = self._initialize_get_collection("relationships")

        # check size
        if relationship_collection.count() >= 1:
            print(
                f"Initializing relationship collection already containing {relationship_collection.count()} entries..."
            )
            return relationship_collection

        # add relationships to collection
        entries = []
        uids = []
        print("Adding relationships to collection...")
        for k, v in tqdm(relationships.items()):
            # replace ddx with differential diagnosis
            if "ddx" in k.lower():
                k = k.replace("ddx", "differential diagnosis")
            # add entry
            entries.append(string_preprocess(k))

            # get origins and targets
            origins = [i for i in v["origins"] if type(i) is str]
            targets = [i for i in v["targets"] if type(i) is str]

            # join to string
            try:
                origins = " ".join(origins) if len(origins) > 0 else " "
            except:
                raise ValueError(f"Failed to join origins: {origins}")

            try:
                targets = " ".join(targets) if len(targets) > 0 else " "
            except:
                raise ValueError(f"Failed to join targets: {targets}")

            # add uids
            uids.append(
                {"origins": origins, "targets": targets}
            )  # already a dict with keys "origins" and "targets" for the uids for valid origin/target concepts and atoms

        # make hashes for id
        hashes = [hash_string(entry) for entry in entries]

        # add entries to collection with uids as metadata, hashes as id for chromadb
        # iterate by self.batch_size to avoid memory issues
        for i in tqdm(range(0, len(entries), self.batch_size)):
            # get batch
            batch = entries[i : i + self.batch_size]
            batch_uids = uids[i : i + self.batch_size]
            batch_hashes = hashes[i : i + self.batch_size]

            try:
                # add batch to collection
                relationship_collection.add(
                    documents=batch,
                    ids=batch_hashes,
                    metadatas=batch_uids,
                )
            except Exception as e:
                print(f"Failed to add batch to collection: {e}")
                raise e

        return relationship_collection

    def match_relationship(self, candidate_relationship: str, k_r: int = 1) -> str:
        """
        Match a relationship.
        """
        # string clean query
        query = string_preprocess(candidate_relationship)

        # retrieve k_r nearest neighbors
        try:
            results = self.relationship_collection.query(
                query_texts=[query],
                n_results=k_r,
                include=["documents", "metadatas"],
            )
        except RuntimeError as e:
            try:
                results = self.relationship_collection.query(
                    query_texts=[query],
                    n_results=1,
                    include=["documents", "metadatas"],
                )
            except Exception as e:
                print(f"Failed to query relationship collection for: {query}")
                raise e
        except Exception as e:
            print(f"Failed to query relationship collection for: {query}")
            raise e

        return results

    def match_concept(
        self,
        concept: str,
        k_c: int = 1,
        uid_str: str = "",
    ) -> str:
        """
        Match a concept.
        """
        # string clean query
        query = string_preprocess(concept)

        # retrieve k_c nearest neighbors
        try:
            if uid_str != "":
                results = self.concept_collection.query(
                    query_texts=[query],
                    n_results=k_c,
                    include=["documents", "distances", "metadatas"],
                    where={"uids": {"$in": uid_str.split(" ")}},
                )
            else:
                results = self.concept_collection.query(
                    query_texts=[query],
                    n_results=k_c,
                    include=["documents", "distances", "metadatas"],
                )
        except RuntimeError as e:
            try:
                if uid_str != "":
                    results = self.concept_collection.query(
                        query_texts=[query],
                        n_results=1,
                        include=["documents", "distances", "metadatas"],
                        where={"uids": {"$in": uid_str.split(" ")}},
                    )
                else:
                    results = self.concept_collection.query(
                        query_texts=[query],
                        n_results=1,
                        include=["documents", "distances", "metadatas"],
                    )
            except Exception as e:
                print(f"Failed to query concept collection for: {query}")
                raise e
        except Exception as e:
            print(f"Failed to query concept collection for: {query}")
            raise e

        return results


# Knowledge Graph class
class KnowledgeGraph:
    """
    Class for biomedical knowledge graph.
    """

    def __init__(self, graph: list[dict]):
        (
            self.edges,
            self.nodes,
            self.relationships,
            self.concepts,
        ) = self.initialize_edges(graph)

    def initialize_edges(self, graph: list[dict]):
        """
        Initialize edges from a graph.
        """
        edge_processed = {}
        nodes = []
        relationships = {}
        concepts = {}
        num_edges = 0

        for edge in graph:
            # split edge
            origin = edge["origin"]
            target = edge["target"]
            relationship = edge["relationship"]

            origin_id = hash_string(origin)
            target_id = hash_string(target)

            # add to edge_processed
            if relationship not in edge_processed:
                edge_processed[relationship] = {}

            if origin not in edge_processed[relationship]:
                edge_processed[relationship][origin] = {}

            if target not in edge_processed[relationship][origin]:
                edge_processed[relationship][origin][target] = True

            # add to nodes
            nodes.append(origin)
            nodes.append(target)

            # add to relationships
            if relationship not in relationships:
                relationships[relationship] = {
                    "origins": [],
                    "targets": [],
                }

            if origin not in relationships[relationship]["origins"]:
                relationships[relationship]["origins"].append(origin_id)

            if target not in relationships[relationship]["targets"]:
                relationships[relationship]["targets"].append(target_id)

            # add to concepts
            if origin not in concepts:
                concepts[origin] = [origin_id]
            else:
                concepts[origin].append(origin_id)

            if target not in concepts:
                concepts[target] = [target_id]
            else:
                concepts[target].append(target_id)

            # increment num_edges
            num_edges += 1

        # Make sure relationships, concepts, and nodes are unique
        for k, v in relationships.items():
            v["origins"] = list(set(v["origins"]))
            v["targets"] = list(set(v["targets"]))

        for k, v in concepts.items():
            concepts[k] = list(set(v))

        nodes = list(set(nodes))

        # Print total edges and return
        print("Total edges: ", num_edges)
        return edge_processed, nodes, relationships, concepts

    def node_exact_match(self, concept: str) -> bool:
        """
        Check if a node exists in the knowledge graph.
        """
        # Check if node exists
        concept = string_preprocess(concept).lower()
        node_exists_result = concept in self.nodes

        # Return result
        return node_exists_result, concept

    def path_exists(self, origin: str, target: str, relationship: str) -> bool:
        """
        Check if a path exists from origin to target via relationship in the knowledge graph.
        """
        # Check if edge exists
        path_exists_result = (
            relationship in self.edges
            and origin in self.edges[relationship]
            and target in self.edges[relationship][origin]
        )

        # Return result
        return path_exists_result

    def query(self, origins: list, targets: list, relationship: str):
        """
        Query the knowledge graph for the existence of a path from origin to target via relationship.
        """
        # Iterate over origins
        for origin in origins:
            # Iterate over targets
            for target in targets:
                # Check if path exists
                if self.path_exists(origin, target, relationship):
                    # Append to valid paths
                    return True, " ".join([origin, relationship, target])

        # Return True if valid paths exist
        return False, ""
